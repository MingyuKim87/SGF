# lshash/lshash.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import int, round, str, object  # noqa

try:
    basestring
except NameError:
    basestring = str

import torch
import operator
import pickle
from collections import defaultdict
# import future        # noqa
import builtins  # noqa
# import past          # noqa
import six  # noqa

import os
import json
import numpy as np

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None

try:
    xrange  # py2
except NameError:
    xrange = range  # py3


class LSHash(object):
    """ LSHash implments locality sensitive hashing using random projection for
    input vectors of dimension `input_dim`.

    Attributes:

    :param hash_size:
        The length of the resulting binary hash in integer. E.g., 32 means the
        resulting binary hash will be 32-bit long.
    :param input_dim:
        The dimension of the input vector. E.g., a grey-scale picture of 30x30
        pixels will have an input dimension of 900.
    :param num_hashtables:
        (optional) The number of hash tables used for multiple lookups.
    :param storage_config:
        (optional) A dictionary of the form `{backend_name: config}` where
        `backend_name` is the either `dict` or `redis`, and `config` is the
        configuration used by the backend. For `redis` it should be in the
        format of `{"redis": {"host": hostname, "port": port_num}}`, where
        `hostname` is normally `localhost` and `port` is normally 6379.
    :param matrices_filename:
        (optional) Specify the path to the compressed numpy file ending with
        extension `.npz`, where the uniform random planes are stored, or to be
        stored if the file does not exist yet.
    :param overwrite:
        (optional) Whether to overwrite the matrices file if it already exist
    """

    def __init__(self, hash_size, input_dim, num_hashtables=1, max_buckets=2, omega=1., data_dim=2,
                 storage_config=None, matrices_filename=None, hashtable_filename=None, overwrite=False, device='cuda:0'):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        self.max_buckets = max_buckets
        self.data_dim = data_dim
        self.device = device
        self.zeros = torch.tensor([0.] * self.data_dim + [0.], device=self.device)

        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = storage_config

        if matrices_filename and not matrices_filename.endswith('.npz'):
            raise ValueError("The specified file name must end with .npz")
        self.matrices_filename = matrices_filename

        if hashtable_filename and not hashtable_filename.endswith('.pickle'):
            raise ValueError("The specified file name must end with .pickle")
        self.hashtable_filename = hashtable_filename

        self.overwrite = overwrite

        self._init_uniform_planes(omega)
        self._init_hashtables()

    def _init_uniform_planes(self, omega=1.):
        """ Initialize uniform planes used to calculate the hashes

        if file `self.matrices_filename` exist and `self.overwrite` is
        selected, save the uniform planes to the specified file.

        if file `self.matrices_filename` exist and `self.overwrite` is not
        selected, load the matrix with `np.load`.

        if file `self.matrices_filename` does not exist and regardless of
        `self.overwrite`, only set `self.uniform_planes`.
        """

        if "uniform_planes" in self.__dict__:
            return

        if self.matrices_filename:
            file_exist = os.path.isfile(self.matrices_filename)
            if file_exist and not self.overwrite:
                try:
                    npzfiles = np.load(self.matrices_filename)
                except IOError:
                    print("Cannot load specified file as a numpy array")
                    raise
                else:
                    npzfiles = sorted(npzfiles.items(), key=lambda x: x[0])
                    self.uniform_planes = [t[1] for t in npzfiles]
            else:
                self.uniform_planes = [self._generate_uniform_planes(omega)
                                       for _ in xrange(self.num_hashtables)]
                try:
                    np.savez_compressed(self.matrices_filename,
                                        *self.uniform_planes)
                except IOError:
                    print("IOError when saving matrices to specificed path")
                    raise
        else:
            self.uniform_planes = [self._generate_uniform_planes(omega)
                                   for _ in xrange(self.num_hashtables)]

    def _init_hashtables(self):
        """ Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """

        if self.hashtable_filename:
            file_exist = os.path.isfile(self.hashtable_filename)
            if file_exist:
                # try:
                with open(self.hashtable_filename, 'rb') as f:
                    data = pickle.load(f)
                self.hash_tables = data['hash_tables']
                self.uniform_planes = data['uniform_planes']
                # npzfiles = np.load(self.hashtable_filename, allow_pickle=True)
                # self.hash_tables = npzfiles['hash_tables']
                # self.uniform_planes = npzfiles['uniform_planes']
            else:
                self.hash_tables = [storage(self.zeros)
                                    for i in xrange(self.num_hashtables)]
        else:
            self.hash_tables = [storage(self.zeros)
                                for i in xrange(self.num_hashtables)]

    def _generate_uniform_planes(self, omega):
        """ Generate uniformly distributed hyperplanes and return it as a 2D
        numpy array.
        """

        return torch.cat((torch.randn(self.hash_size, self.input_dim, device=self.device) / omega, torch.rand(self.hash_size, 1, device=self.device)), 1)

    def _hash(self, planes, input_point):
        """ Generates the binary hash for `input_point` and returns it.

        :param planes:
            The planes are random uniform planes with a dimension of
            `hash_size` * `input_dim`.
        :param input_point:
            A Python tuple or list object that contains only numbers.
            The dimension needs to be 1 * `input_dim`.
        """
        projections = torch.mm(planes[:,:-1], input_point) + planes[:,-1:]
        print(projections[:,0])
        projections = (torch.floor(projections).int().t() % self.max_buckets).cpu().numpy().astype(str)
        out = [''.join(row) for row in projections]
        return out

    def index(self, input_point, feature=[], retrieve_samples=False):
        for i, table in enumerate(self.hash_tables):
            hs = self._hash(self.uniform_planes[i], input_point[:-1] if len(feature) == 0 else feature)
            # print(hs)
            for idx, h in enumerate(hs):
                table.storage[h] = table.storage.get(h, self.zeros) + input_point[:,idx]
                # table.storage[h] = table.storage[h] + input_point[:, idx]
                # table.storage[h] = [sum(x) for x in zip(table.storage.get(h, zeros), input_point + [1])]
            # getter = operator.itemgetter(*h)
            # values = getter(table.storage)
            # result = self.safe_itemgetter(h, table.storage, zeros)()
            # print(result)
            # import sys
            # sys.exit()
            # table.storage[h] = [sum(x) for x in zip(table.storage.get(h, zeros), input_point + [1])]

    def query(self, query_point, retrieve_samples=False, distance_func=None):
        value = torch.tensor([0.] * self.data_dim + [0.], device=self.device)
        for i, table in enumerate(self.hash_tables):
            hs = self._hash(self.uniform_planes[i], query_point)
            for idx, h in enumerate(hs):
                value = value + table.storage.get(h, self.zeros)
                # value = value + table.storage[h]

        return value

    def safe_itemgetter(self, keys, dictionary, default=None):
        getter = operator.itemgetter(*keys)
        def wrapper():
            try:
                return getter(dictionary)
            except KeyError:
                return [dictionary.get(key, default) for key in keys]
        return wrapper

    def save(self, filename):
        """
            Save save the uniform planes to the specified file.
        """
        # if filename:
        #     try:
        with open(filename, 'wb') as f:
            pickle.dump({'hash_tables': self.hash_tables, 'uniform_planes': self.uniform_planes}, f)

            #     # np.savez_compressed(filename, allow_pickle=True, hash_tables=self.hash_tables, uniform_planes=self.uniform_planes)
            #
            # except IOError:
            #     print("IOError when saving hash tables to specificed path")
            #     raise



    def get_hashes(self, input_point):
        """ Takes a single input point `input_point`, iterate through the
        uniform planes, and returns a list with size of `num_hashtables`
        containing the corresponding hash for each hashtable.

        :param input_point:
            A list, or tuple, or numpy ndarray object that contains numbers
            only. The dimension needs to be 1 * `input_dim`.
        """

        hashes = []
        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(self.uniform_planes[i], input_point)
            hashes.append(binary_hash)

        return hashes

    ### distance functions

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)


def storage(zeros):
    """ Given the configuration for storage and the index, return the
    configured storage instance.
    """
    return InMemoryStorage(zeros)


class BaseStorage(object):
    def __init__(self, config):
        """ An abstract class used as an adapter for storages. """
        raise NotImplementedError

    def keys(self):
        """ Returns a list of binary hashes that are used as dict keys. """
        raise NotImplementedError

    def set_val(self, key, val):
        """ Set `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def get_val(self, key):
        """ Return `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def append_val(self, key, val):
        """ Append `val` to the list stored at `key`.

        If the key is not yet present in storage, create a list with `val` at
        `key`.
        """
        raise NotImplementedError

    def get_list(self, key):
        """ Returns a list stored in storage at `key`.

        This method should return a list of values stored at `key`. `[]` should
        be returned if the list is empty or if `key` is not present in storage.
        """
        raise NotImplementedError


class InMemoryStorage(BaseStorage):
    def __init__(self, default_value):
        self.name = 'dict'
        # self.storage = defaultdict(lambda: default_value)
        self.storage = dict()

    def keys(self):
        return self.storage.keys()

    def set_val(self, key, val):
        self.storage[key] = val

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val):
        self.storage.setdefault(key, []).append(val)

    def get_list(self, key):
        return self.storage.get(key, [])