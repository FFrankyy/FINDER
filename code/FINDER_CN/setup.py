
from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},

    #################for ubuntu compile
    ext_modules = [
                    Extension('PrepareBatchGraph', sources = ['./FINDER_CN/PrepareBatchGraph.pyx','./FINDER_CN/src/lib/PrepareBatchGraph.cpp','./FINDER_CN/src/lib/graph.cpp','./FINDER_CN/src/lib/graph_struct.cpp',  './FINDER_CN/src/lib/disjoint_set.cpp'],language='c++',extra_compile_args=['-std=c++11']),
                   Extension('graph', sources=['./FINDER_CN/graph.pyx', './FINDER_CN/src/lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('mvc_env', sources=['./FINDER_CN/mvc_env.pyx', './FINDER_CN/src/lib/mvc_env.cpp', './FINDER_CN/src/lib/graph.cpp','./FINDER_CN/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('utils', sources=['./FINDER_CN/utils.pyx', './FINDER_CN/src/lib/utils.cpp', './FINDER_CN/src/lib/graph.cpp', './FINDER_CN/src/lib/graph_utils.cpp', './FINDER_CN/src/lib/disjoint_set.cpp', './FINDER_CN/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem', sources=['./FINDER_CN/nstep_replay_mem.pyx', './FINDER_CN/src/lib/nstep_replay_mem.cpp', './FINDER_CN/src/lib/graph.cpp', './FINDER_CN/src/lib/mvc_env.cpp', './FINDER_CN/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem_prioritized',sources=['./FINDER_CN/nstep_replay_mem_prioritized.pyx', './FINDER_CN/src/lib/nstep_replay_mem_prioritized.cpp','./FINDER_CN/src/lib/graph.cpp', './FINDER_CN/src/lib/mvc_env.cpp', './FINDER_CN/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['./FINDER_CN/graph_struct.pyx', './FINDER_CN/src/lib/graph_struct.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('GraphDQN', sources = ['./FINDER_CN/GraphDQN.pyx'])
                   ])