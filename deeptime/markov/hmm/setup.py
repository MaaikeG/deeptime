# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('hmm', parent_package, top_path)

    config.add_subpackage("init")

    config.add_extension('_hmm_bindings',
                         sources=['_bindings/src/hmm_module.cpp'],
                         include_dirs=['_bindings/include'],
                         language='c++',
                         extra_compile_args=['-fvisibility=hidden']
                         )

    return config
