import subprocess
import os


def main():
    energyplus_to_fmu_path = r'energyplus_to_fmu\EnergyPlusToFMU.py'
    model_file = r'OverhangTest241104.idf'  # 使用绝对路径
    idd_file = r'C:\EnergyPlusV24-1-0\Energy+.idd'
    epw_file = r'..\source\epw_file\CHN_Guangdong.Shenzhen.594930_SWERA.epw'

    # 检查文件是否存在
    if not all(os.path.exists(f) for f in [energyplus_to_fmu_path, model_file, idd_file, epw_file]):
        print("Error: One or more files do not exist.")
        return

    # 构建命令字符串
    convert_command = f"python {energyplus_to_fmu_path} -d -i {idd_file} -a 2 -w {epw_file} {model_file}"

    # 执行转换命令
    try:
        subprocess.check_call(convert_command, shell=True)
        print("模型转换为FMU完成。")
    except subprocess.CalledProcessError as e:
        print("转换过程出错:", e)
        print("详细错误信息:", e.output)


if __name__ == '__main__':
    main()
