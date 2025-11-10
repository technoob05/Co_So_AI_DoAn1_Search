"""
Module quản lý cấu hình (Configuration Management)

Chức năng:
- Load/Save configurations từ YAML/JSON
- Validate parameters
- Default configurations cho mỗi thuật toán
"""

import yaml
import json
import os
from typing import Dict, Any, List
import numpy as np


class ConfigManager:
    """
    Class quản lý configurations
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Khởi tạo ConfigManager
        
        Parameters:
        -----------
        config_file : str
            Đường dẫn file config
        """
        self.config_file = config_file
        self.config = {}
        
        # Load config nếu file tồn tại
        if os.path.exists(config_file):
            self.load_config(config_file)
        else:
            # Sử dụng default config
            self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Trả về default configuration
        
        Returns:
        --------
        config : dict
            Default configuration
        """
        config = {
            # Swarm Intelligence Algorithms
            'algorithms': {
                'PSO': {
                    'name': 'Particle Swarm Optimization',
                    'type': 'swarm',
                    'default_params': {
                        'n_particles': 30,
                        'max_iter': 100,
                        'w': 0.7,
                        'c1': 1.5,
                        'c2': 1.5
                    },
                    'param_ranges': {
                        'n_particles': [10, 100],
                        'max_iter': [10, 500],
                        'w': [0.1, 1.0],
                        'c1': [0.5, 3.0],
                        'c2': [0.5, 3.0]
                    },
                    'description': 'Mô phỏng hành vi bầy đàn của chim và cá'
                },
                'ACO': {
                    'name': 'Ant Colony Optimization',
                    'type': 'swarm',
                    'default_params': {
                        'n_ants': 30,
                        'max_iter': 100,
                        'archive_size': 50,
                        'q': 0.5,
                        'xi': 0.85
                    },
                    'param_ranges': {
                        'n_ants': [10, 100],
                        'max_iter': [10, 500],
                        'archive_size': [20, 100],
                        'q': [0.1, 1.0],
                        'xi': [0.5, 1.0]
                    },
                    'description': 'Lấy cảm hứng từ kiến tìm đường bằng pheromone'
                },
                'ABC': {
                    'name': 'Artificial Bee Colony',
                    'type': 'swarm',
                    'default_params': {
                        'n_bees': 30,
                        'max_iter': 100,
                        'limit': 20
                    },
                    'param_ranges': {
                        'n_bees': [10, 100],
                        'max_iter': [10, 500],
                        'limit': [10, 50]
                    },
                    'description': 'Mô phỏng hành vi tìm thức ăn của ong mật'
                },
                'Firefly': {
                    'name': 'Firefly Algorithm',
                    'type': 'swarm',
                    'default_params': {
                        'n_fireflies': 30,
                        'max_iter': 100,
                        'alpha': 0.5,
                        'beta0': 1.0,
                        'gamma': 1.0
                    },
                    'param_ranges': {
                        'n_fireflies': [10, 100],
                        'max_iter': [10, 500],
                        'alpha': [0.1, 1.0],
                        'beta0': [0.5, 2.0],
                        'gamma': [0.1, 2.0]
                    },
                    'description': 'Dựa trên ánh sáng của đom đóm'
                },
                'Cuckoo': {
                    'name': 'Cuckoo Search',
                    'type': 'swarm',
                    'default_params': {
                        'n_nests': 30,
                        'max_iter': 100,
                        'pa': 0.25,
                        'beta': 1.5
                    },
                    'param_ranges': {
                        'n_nests': [10, 100],
                        'max_iter': [10, 500],
                        'pa': [0.1, 0.5],
                        'beta': [1.0, 3.0]
                    },
                    'description': 'Dựa trên hành vi đẻ trứng ký sinh của chim cu-cu'
                },
                
                # Traditional Search Algorithms
                'HillClimbing': {
                    'name': 'Hill Climbing',
                    'type': 'traditional',
                    'default_params': {
                        'max_iter': 100,
                        'step_size': 0.5
                    },
                    'param_ranges': {
                        'max_iter': [10, 500],
                        'step_size': [0.1, 2.0]
                    },
                    'description': 'Thuật toán tìm kiếm cục bộ đơn giản'
                },
                'SimulatedAnnealing': {
                    'name': 'Simulated Annealing',
                    'type': 'traditional',
                    'default_params': {
                        'max_iter': 1000,
                        'initial_temp': 100.0,
                        'final_temp': 0.001,
                        'alpha': 0.95,
                        'step_size': 1.0
                    },
                    'param_ranges': {
                        'max_iter': [100, 5000],
                        'initial_temp': [50.0, 200.0],
                        'final_temp': [0.0001, 0.01],
                        'alpha': [0.8, 0.99],
                        'step_size': [0.1, 2.0]
                    },
                    'description': 'Mô phỏng quá trình ủ kim loại'
                },
                'GeneticAlgorithm': {
                    'name': 'Genetic Algorithm',
                    'type': 'traditional',
                    'default_params': {
                        'population_size': 50,
                        'max_iter': 100,
                        'crossover_rate': 0.8,
                        'mutation_rate': 0.1,
                        'elite_size': 2
                    },
                    'param_ranges': {
                        'population_size': [10, 100],
                        'max_iter': [10, 500],
                        'crossover_rate': [0.5, 1.0],
                        'mutation_rate': [0.01, 0.3],
                        'elite_size': [1, 10]
                    },
                    'description': 'Mô phỏng quá trình tiến hóa tự nhiên'
                },
                
                # Graph Search Algorithms
                'BFS': {
                    'name': 'Breadth-First Search',
                    'type': 'graph',
                    'default_params': {},
                    'param_ranges': {},
                    'description': 'Tìm kiếm theo chiều rộng, đảm bảo đường đi ngắn nhất'
                },
                'DFS': {
                    'name': 'Depth-First Search',
                    'type': 'graph',
                    'default_params': {
                        'max_depth': None
                    },
                    'param_ranges': {
                        'max_depth': [10, 1000]
                    },
                    'description': 'Tìm kiếm theo chiều sâu'
                },
                'AStar': {
                    'name': 'A* Search',
                    'type': 'graph',
                    'default_params': {
                        'heuristic': 'manhattan'
                    },
                    'param_ranges': {},
                    'description': 'Tìm kiếm có thông tin với heuristic'
                }
            },
            
            # Test Functions
            'test_functions': {
                'sphere': {
                    'name': 'Sphere Function',
                    'bounds': [-100, 100],
                    'global_optimum': 0.0,
                    'difficulty': 'easy',
                    'properties': ['unimodal', 'convex', 'separable']
                },
                'rastrigin': {
                    'name': 'Rastrigin Function',
                    'bounds': [-5.12, 5.12],
                    'global_optimum': 0.0,
                    'difficulty': 'hard',
                    'properties': ['multimodal', 'separable']
                },
                'rosenbrock': {
                    'name': 'Rosenbrock Function',
                    'bounds': [-5, 10],
                    'global_optimum': 0.0,
                    'difficulty': 'medium',
                    'properties': ['unimodal', 'non-convex', 'non-separable']
                },
                'ackley': {
                    'name': 'Ackley Function',
                    'bounds': [-32.768, 32.768],
                    'global_optimum': 0.0,
                    'difficulty': 'hard',
                    'properties': ['multimodal', 'non-separable']
                }
            },
            
            # Discrete Problems
            'discrete_problems': {
                'tsp': {
                    'name': 'Traveling Salesman Problem',
                    'default_params': {
                        'n_cities': 20
                    }
                },
                'knapsack': {
                    'name': '0/1 Knapsack Problem',
                    'default_params': {
                        'n_items': 20,
                        'capacity': None  # Auto-calculate
                    }
                },
                'graph_coloring': {
                    'name': 'Graph Coloring Problem',
                    'default_params': {
                        'n_vertices': 20,
                        'edge_probability': 0.3
                    }
                }
            },
            
            # Visualization Settings
            'visualization': {
                'plot_style': 'seaborn',
                'figure_dpi': 300,
                'figure_formats': ['png', 'pdf'],
                'default_figsize': [12, 8],
                'color_palette': 'Set3',
                'animation_fps': 10
            },
            
            # Experiment Settings
            'experiments': {
                'default_n_runs': 30,
                'default_n_jobs': -1,  # -1 = use all cores
                'random_seed': 42,
                'convergence_threshold': 0.01,
                'max_time_per_run': 300  # seconds
            },
            
            # Logging Settings
            'logging': {
                'log_dir': 'logs',
                'results_dir': 'results',
                'save_plots': True,
                'save_data': True,
                'verbose': True
            }
        }
        
        return config
    
    def load_config(self, config_file: str):
        """
        Load configuration từ file
        
        Parameters:
        -----------
        config_file : str
            Đường dẫn file config (YAML hoặc JSON)
        """
        self.config_file = config_file
        
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config_file.endswith('.json'):
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file}")
    
    def save_config(self, config_file: str = None):
        """
        Lưu configuration vào file
        
        Parameters:
        -----------
        config_file : str, optional
            Đường dẫn file để lưu (nếu None thì dùng self.config_file)
        """
        if config_file is None:
            config_file = self.config_file
        
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        elif config_file.endswith('.json'):
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_file}")
    
    def get_algorithm_params(self, algorithm: str, custom_params: Dict = None) -> Dict:
        """
        Lấy parameters cho thuật toán (kết hợp default và custom)
        
        Parameters:
        -----------
        algorithm : str
            Tên thuật toán
        custom_params : dict, optional
            Custom parameters để override defaults
            
        Returns:
        --------
        params : dict
            Parameters đầy đủ
        """
        if algorithm not in self.config['algorithms']:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Lấy default params
        params = self.config['algorithms'][algorithm]['default_params'].copy()
        
        # Override với custom params
        if custom_params:
            params.update(custom_params)
        
        # Validate params
        self.validate_params(algorithm, params)
        
        return params
    
    def validate_params(self, algorithm: str, params: Dict) -> bool:
        """
        Validate parameters nằm trong range hợp lệ
        
        Parameters:
        -----------
        algorithm : str
            Tên thuật toán
        params : dict
            Parameters cần validate
            
        Returns:
        --------
        valid : bool
            True nếu hợp lệ
            
        Raises:
        -------
        ValueError
            Nếu parameters không hợp lệ
        """
        if algorithm not in self.config['algorithms']:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        param_ranges = self.config['algorithms'][algorithm].get('param_ranges', {})
        
        for param_name, param_value in params.items():
            if param_name in param_ranges:
                min_val, max_val = param_ranges[param_name]
                
                if param_value is not None:
                    if not (min_val <= param_value <= max_val):
                        raise ValueError(
                            f"Parameter {param_name}={param_value} out of range "
                            f"[{min_val}, {max_val}] for algorithm {algorithm}"
                        )
        
        return True
    
    def get_test_function_info(self, function_name: str) -> Dict:
        """
        Lấy thông tin về test function
        
        Parameters:
        -----------
        function_name : str
            Tên test function
            
        Returns:
        --------
        info : dict
            Thông tin về function
        """
        if function_name not in self.config['test_functions']:
            raise ValueError(f"Unknown test function: {function_name}")
        
        return self.config['test_functions'][function_name]
    
    def get_all_algorithms(self, algo_type: str = None) -> List[str]:
        """
        Lấy danh sách tất cả thuật toán
        
        Parameters:
        -----------
        algo_type : str, optional
            Filter theo loại ('swarm', 'traditional', 'graph')
            
        Returns:
        --------
        algorithms : list
            Danh sách tên thuật toán
        """
        if algo_type is None:
            return list(self.config['algorithms'].keys())
        else:
            return [name for name, info in self.config['algorithms'].items()
                   if info['type'] == algo_type]
    
    def get_visualization_settings(self) -> Dict:
        """
        Lấy visualization settings
        
        Returns:
        --------
        settings : dict
            Visualization settings
        """
        return self.config.get('visualization', {})
    
    def get_experiment_settings(self) -> Dict:
        """
        Lấy experiment settings
        
        Returns:
        --------
        settings : dict
            Experiment settings
        """
        return self.config.get('experiments', {})


if __name__ == "__main__":
    # Test ConfigManager
    print("Testing ConfigManager...")
    print("=" * 60)
    
    # Khởi tạo với default config
    config_manager = ConfigManager()
    
    # Lưu default config ra file
    config_manager.save_config("config.yaml")
    print("Saved default config to config.yaml")
    
    # Test lấy algorithm params
    pso_params = config_manager.get_algorithm_params('PSO')
    print(f"\nPSO default params: {pso_params}")
    
    # Test lấy params với custom values
    custom_params = config_manager.get_algorithm_params('PSO', {'n_particles': 50, 'w': 0.8})
    print(f"PSO custom params: {custom_params}")
    
    # Test validate params
    try:
        config_manager.validate_params('PSO', {'w': 1.5})  # Valid
        print("✓ Validation passed for valid params")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    try:
        config_manager.validate_params('PSO', {'w': 2.0})  # Invalid (out of range)
        print("✗ Validation should have failed!")
    except ValueError as e:
        print(f"✓ Validation correctly failed: {e}")
    
    # Test lấy thông tin test function
    sphere_info = config_manager.get_test_function_info('sphere')
    print(f"\nSphere function info: {sphere_info}")
    
    # Test lấy danh sách thuật toán
    all_algos = config_manager.get_all_algorithms()
    print(f"\nAll algorithms: {all_algos}")
    
    swarm_algos = config_manager.get_all_algorithms('swarm')
    print(f"Swarm algorithms: {swarm_algos}")
    
    print("\nTest completed successfully!")

