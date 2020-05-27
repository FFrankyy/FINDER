from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},

    #################for ubuntu compile
    ext_modules = [
                    Extension('PrepareBatchGraph', sources = ['./FINDER_ND_cost/PrepareBatchGraph.pyx','./FINDER_ND_cost/src/lib/PrepareBatchGraph.cpp','./FINDER_ND_cost/src/lib/graph.cpp','./FINDER_ND_cost/src/lib/graph_struct.cpp',  './FINDER_ND_cost/src/lib/disjoint_set.cpp'],language='c++',extra_compile_args=['-std=c++11']),
                   Extension('graph', sources=['./FINDER_ND_cost/graph.pyx', './FINDER_ND_cost/src/lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('mvc_env', sources=['./FINDER_ND_cost/mvc_env.pyx', './FINDER_ND_cost/src/lib/mvc_env.cpp', './FINDER_ND_cost/src/lib/graph.cpp','./FINDER_ND_cost/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('utils', sources=['./FINDER_ND_cost/utils.pyx', './FINDER_ND_cost/src/lib/utils.cpp', './FINDER_ND_cost/src/lib/graph.cpp', './FINDER_ND_cost/src/lib/graph_utils.cpp', './FINDER_ND_cost/src/lib/disjoint_set.cpp', './FINDER_ND_cost/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem', sources=['./FINDER_ND_cost/nstep_replay_mem.pyx', './FINDER_ND_cost/src/lib/nstep_replay_mem.cpp', './FINDER_ND_cost/src/lib/graph.cpp', './FINDER_ND_cost/src/lib/mvc_env.cpp', './FINDER_ND_cost/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem_prioritized',sources=['./FINDER_ND_cost/nstep_replay_mem_prioritized.pyx', './FINDER_ND_cost/src/lib/nstep_replay_mem_prioritized.cpp','./FINDER_ND_cost/src/lib/graph.cpp', './FINDER_ND_cost/src/lib/mvc_env.cpp', './FINDER_ND_cost/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['./FINDER_ND_cost/graph_struct.pyx', './FINDER_ND_cost/src/lib/graph_struct.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('GraphDQN', sources = ['./FINDER_ND_cost/GraphDQN.pyx'])
                   ])

