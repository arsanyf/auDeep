# Copyright (C) 2017-2018 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.

"""Parser for the Bipolar data set"""
from pathlib import Path
from typing import Mapping, Sequence
from sklearn.model_selection import StratifiedKFold

import pandas as pd

from audeep.backend.data.data_set import Split
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata

_LABEL_MAP = {
    "0": 0,
    "1": 1
}


class BipParser(LoggingMixin, Parser):
    """
    Parser for the Bipolar data set.
    """

    def __init__(self, basedir: Path):
        """
        Creates and initializes a new BipParser for the specified base directory.
        
        Parameters
        ----------
        basedir: pathlib.Path
            The data set base directory
        """
        super().__init__(basedir)

        self._metadata_cache = None
        self._cv_setup_cache = None

    def _metadata(self) -> pd.DataFrame:
        """
        Read the Bipolar_metadata_anno.csv file in the data set base directory containing general data set metadata.
        
        The Bipolar_metadata_anno.csv file is read only once and cached.
        
        Returns
        -------
        pandas.DataFrame
            The metadata contained in the Bipolar_metadata_anno.csv file as a pandas DataFrame
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse Bip dataset at {}".format(self._basedir))
        if self._metadata_cache is None:
            self._metadata_cache = pd.read_csv(str(self._basedir / "Bipolar_metadata_anno.csv"),delimiter=";", header=0, usecols=['IsManic', 'WaveFilename'])

        # noinspection PyTypeChecker
        return self._metadata_cache

    def _cv_setup(self) -> Sequence[pd.DataFrame]:
        """
        Setup cross validation using StratifiedKFold
        
        Returns
        -------
        list of pandas.DataFrame
            A list of pandas DataFrames containing the contents of the fold
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse Bip dataset at {}".format(self._basedir))
        if self._cv_setup_cache is None:
            skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True)
            self._cv_setup_cache = []
            for train, test in skf.split(a['WaveFilename'], a['IsManic']):
                self._cv_setup_cache.append(self._metadata_cache.iloc[[train]])

        return self._cv_setup_cache

    @property
    def num_instances(self) -> int:
        """
        Returns the number of instances in the data set.
        
        Returns
        -------
        int
            The number of instances in the data set
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        return len(self._metadata())

    @property
    def num_folds(self) -> int:
        """
        Returns the number of cross-validation folds, which is four for this parser.
        
        Returns
        -------
        int
            Four
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        return 4

    @property
    def label_map(self) -> Mapping[str, int]:
        """
        Returns the mapping of nominal to numeric labels.
        
        Nominal labels are assigned integer indices in alphabetical order. That is, the following label map is returned:
        
        "0": 0,
        "1": 1,
        
        Returns
        -------
        map of str to int
            The mapping of nominal to numeric labels
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        return _LABEL_MAP

    def can_parse(self) -> bool:
        """
        Checks whether the data set base directory contains the Bip data set.
        
        Currently, this method checks for the presence of a Bipolar_metadata_anno.csv file
        in the data set base directory, and the presence of Kontrol and Mani directories
        
        Returns
        -------
        bool
             True, if the data set base directory contains the Bip data set, False otherwise
        """
        anno_exists = (self._basedir / "Bipolar_metadata_anno.csv").exists()
        kontrol_dir_exists = (self._basedir / "Kontrol").exists()
        mani_dir_exists = (self._basedir / "Mani").exists()

        return anno_exists and kontrol_dir_exists and mani_dir_exists

    def parse(self) -> Sequence[_InstanceMetadata]:
        """
        Parses the instances contained in this data set.

        For each instance, metadata is computed and stored in an _InstanceMetadata object. Instances are parsed in the
        order in which they appear in the meta.txt file.

        Returns
        -------
        list of _InstanceMetadata
            A list of _InstanceMetadata containing one entry for each parsed audio file
        
        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse DCASE dataset at {}".format(self._basedir))

        metadata = self._metadata()
        cv_setup = self._cv_setup()

        meta_list = []

        for index, row in metadata.iterrows():
            filename = row['WaveFilename']
            label_nominal = row['IsManic']

            if label_nominal not in _LABEL_MAP:
                raise IOError("invalid label for Bip data: {}".format(label_nominal))

            cv_folds = []

            for fold_metadata in cv_setup:
                cv_folds.append(Split.TRAIN if filename in fold_metadata.iloc[:, 0].values else Split.VALID)

            instance_metadata = _InstanceMetadata(path=self._basedir / filename,
                                                  filename=str(Path(filename).name),
                                                  label_nominal=label_nominal,
                                                  label_numeric=None,
                                                  cv_folds=cv_folds,
                                                  partition=None)

            self.log.debug("parsed instance %s: label = %s", filename, label_nominal)
            meta_list.append(instance_metadata)

        return meta_list
