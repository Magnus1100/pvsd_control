from pyfmi import load_fmu
import numpy as np
import matplotlib.pyplot as plt

# 1. 初始化参数
modelname = 'idf_test0901'  # fmu文件

# 耦合模拟计算范围
days = 10
hours = 24
minutes = 60
seconds = 60

EPTtimestep = 6  # timestep 时间步长，单位：min
numsteps = days * hours * EPTtimestep  # 总时长
timestop = days * hours * minutes * seconds  # 模拟总时间
secondstep = timestop / numsteps  # 每个时间步长的时间长度

# 2. 加载FMU模型
model = load_fmu(modelname + '.fmu')  # 读取模型
opts = model.simulate_options()  # 获取仿真选项
opts['ncp'] = numsteps  # 设置仿真步数（ncp 代表 number of communication points）。
opts['initialize'] = False  # 设置是否初始化模型

# 3. 初始化模型
sim_time = 0
# 打印初始参数
print(f"Simulation time: {sim_time}")
print(f"Time stop: {timestop}")

# 运行初始化并捕获日志
try:
    model.initialize(sim_time, timestop)
except Exception as e:
    print(f"Initialization failed: {e}")
    log = model.get_log()
    print("Model log:")
    print(log)

# 4. 创建输入和输出数据
t = np.linspace(0.0, timestop, numsteps)
T_upper = (t * 0.000001) + 23
T_lower = (t * 0.000001) + 21
cooling_setpoint = np.transpose(np.vstack((t, T_upper)))
heating_setpoint = np.transpose(np.vstack((t, T_lower)))
input_object_cooling = ('Tcsetpoint', cooling_setpoint)
input_object_heating = ('Tcsetpoint', heating_setpoint)

# 5. 初始化存储数组
index = 0
inputcheck_cooling = []
inputcheck_heating = []
T1 = np.zeros((numsteps, 1))
T2 = np.zeros((numsteps, 1))
T3 = np.zeros((numsteps, 1))
T4 = np.zeros((numsteps, 1))
Tcore = np.zeros((numsteps, 1))

# 6. 仿真循环
while sim_time < timestop:
    model.set('Tcsetpoint', input_object_cooling)[1][index][1]
    model.set('Thsetpoint', input_object_heating)[1][index][1]
    res = model.do_step(current_t=sim_time, step_size=secondstep, new_step=True)
    inputcheck_cooling[index] = model.get('Tcsetpoint')
    inputcheck_heating[index] = model.get('Thsetpoint')
    T1[index] = model.get('T1')
    T2[index] = model.get('T2')
    T3[index] = model.get('T3')
    T4[index] = model.get('T4')
    Tcore[index] = model.get('Tcore')
    sim_time += secondstep
    index += 1

# 7. 绘图
plt.figure(1)
plt.plot(t, T1, 'b', label='TZone1')
