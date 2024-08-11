import pygmo as pg
import pandas as pd


# 定义一个简单的自定义问题
class MyProblem:
    def __init__(self):
        self.dim = 2  # 问题的维度

    def fitness(self, x):
        # 简单的二次函数作为目标
        return [x[0] ** 2 + x[1] ** 2]

    def get_bounds(self):
        # 决策变量的上下界
        return [-5] * self.dim, [5] * self.dim

    @staticmethod
    def get_name():
        return "My Custom Problem"

    @staticmethod
    def get_nobj():
        return 1  # 单目标问题


# 实例化问题
prob = pg.problem(MyProblem())

# 使用PSO算法进行优化
algo = pg.algorithm(pg.pso(gen=100))
pop_size = 10
isl = pg.island(algo=algo, prob=prob, size=pop_size)

# 执行演化
isl.evolve()
isl.wait_check()

# 获取种群
pop = isl.get_population()

# 获取所有个体的决策变量和适应度
x = pop.get_x()
f = pop.get_f()

# 将个体和适应度组合成一个DataFrame
data = pd.DataFrame(x, columns=['var1', 'var2'])
data['fitness'] = f

# 找到最小适应度
min_fitness = min(f)

# 筛选出适应度等于最优解的所有个体
best_individuals = data[data['fitness'] == min_fitness]

# 打印结果
print("All individuals with the best fitness:")
print(best_individuals)
