import os
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from matplotlib import patches

from GBG_930 import get_gb_division_x
from Filter_930 import filter

warnings.filterwarnings("ignore")
from sklearn.cluster import DBSCAN,KMeans
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "KaiTi"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------------------------------5
# -------------------------------------------------------5
# -------------------------------------------------------5
def json2csv(json_file_path):
    """
    从 JSON 文件提取数据，转换格式为 CSV

    参数:
        json_file_path (str): 输入 JSON 文件路径
    """
    try:
        # 1. 读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)


        # 3. 创建 DataFrame
        data_list = data.get('data', [])  # 获取 data 数组，如果不存在则默认为空列表
        df = pd.DataFrame(data_list)
        df = df[['lat','lng']].astype('float')
        print(df.head())
        if len(df.lat.unique()) == 1:
            print('数据集不符，', json_file_path[-27:], '数据集只包含一个数据点')
            return None

        print(f"成功！共提取 {len(df)} 条记录")
        return df

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None

def plot_df(result_df, plottype,title):
    print('去单线去重后数据量：', len(result_df))
    # print(result_df.head(3))
    plt.figure(figsize=(10, 6), dpi=100)
    plt.title(title)

    center_lat = result_df.iloc[:, 0].mean()
    center_lng = result_df.iloc[:, 1].mean()
    plt.text(center_lng, center_lat, s='center')

    if plottype:
        plt.plot(result_df.iloc[:, 0], result_df.iloc[:, 1], c='#00D000')  # 折线图
    else:
        plt.scatter(result_df.iloc[:, 0], result_df.iloc[:, 1], edgecolors=None,c='#00D000')  # 散点图

    plt.text(result_df.iloc[0, 0], result_df.iloc[0, 1], s='start')
    plt.text(result_df.iloc[-1, 0], result_df.iloc[-1, 1], s='end')

    plt.show()


def plot_csv(df, plottype, size,title):
    # print('数据点个数：', len(df))
    # 如看前几行数据
    # if df is not None:
    #     print("\n数据预览：")
    #     print(df.head(3))

    plt.figure(figsize=(10, 6), dpi=100)
    plt.title(title)

    if plottype:
        plt.plot(df.iloc[:, 0], df.iloc[:, 1])  # 折线图
    else:
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], edgecolors=None, s=size)  # 散点图

    plt.text(df.iloc[0, 0], df.iloc[0, 1], "start")
    plt.text(df.iloc[-1, 0], df.iloc[-1, 1], "end")
    plt.scatter(df.iloc[0, 0], df.iloc[0, 1], s=5, c='g')
    plt.scatter(df.iloc[-1, 0], df.iloc[-1, 1], s=5, c='r')
    plt.show()

# eps=0.000001, min_samples=68


def json2csv_main(df,visual,plot_type,file_name):
    # print(path)

    df = df[['lat','lng']]
    print('原始dflen',len(df))
    # plot_type = 1
    # print(df.head(3))
    if df is not None:
        print('------------------------')
        if visual:
            plot_csv(df, plot_type, 0.5, file_name)


        df,lens = filter(df=df)

        print('------------------------')
        if visual:
            plot_df(df, plot_type, file_name)

        return df, lens

    else:
        print(file_name, '文件不符合要求！！！')
        return 0





# -------------------------------------------------------4
# -------------------------------------------------------4
# -------------------------------------------------------4

def plot_dot(data, title):
    plt.figure(figsize=(10, 8))
    # plt.scatter(data[:, 0], data[:, 1], c="#00D000", alpha=0.8, label='data point')
    plt.plot(data[:, 0], data[:, 1], c='#00D000', label='data point')
    plt.title(title)
    plt.legend()



# -------------------------------------------------------2
# -------------------------------------------------------2
# -------------------------------------------------------2
def divide_ball_GBC_y(X,K,visual,title):
    '''
    先根据GBC中的粒球划分生成粒球，然后按照粒球的最近邻优先的距离度量连接粒球。
    Args:
        X: 数据
        K: 聚类簇数
        detaile: 聚类细节
    Returns:
    '''

    areas = get_gb_division_x(X, True, visual,title)
    del X, K, visual, title
    # --------------------------------------------------------------------------------------------
    return areas


# -------------------------------------------------------1
# -------------------------------------------------------1
# -------------------------------------------------------1
# 2026-3-30 新增的功能，多区域只显示局部作业区域，便于展示粒球可视化

def find_max_diff(df):
    # print('find_max_diff -1',df.head())

    df = df.sort_values(by='lat', ascending=False, ignore_index=True)
    # 计算相邻行的纬度差值（排除首行NaN）
    lat_diff = df['lat'].diff().dropna()
    df['diff'] = pd.Series(lat_diff)
    # 找到最大差值及其位置
    lat_max_gap_value = lat_diff.min()
    lat_diff_mean = lat_diff.mean()

    # print(f'lat_diff.mean(){lat_diff_mean:.6f}')
    # print(f'lat_diff.min(){lat_diff.min():.6f}')
    # print(f'lat_diff.max(){lat_diff.max():.6f}')

    lat_max_gap_positions = lat_diff[lat_diff == lat_max_gap_value].index.tolist()
    lat_max_gap_positions = lat_max_gap_positions[0]
    # print('lat_max_gap_positions', lat_max_gap_positions)
    if lat_max_gap_positions < 6:
        df = df.iloc[lat_max_gap_positions + 1:,:]
    else:
        # print('lat_max_gap_positions', lat_max_gap_positions)

        df['lat_is_max_gap'] = False

        df.loc[lat_max_gap_positions, 'lat_is_max_gap'] = True
    # print(df)
    k = None
    for i in range(len(df)):
        # print(i)
        if df.iloc[i]['diff'] < 10 * lat_diff_mean:
            k = i
            print('k=', k)
            if k < 6:
                print('continue is yes')
                continue

            print('break is yes')
            break
    if k is not None:
        # k -= 2
        df = df.iloc[:k, :]

    # print('find_max_diff -2', df.head())

    return df

def dataframe_to_json(code, area, df0,df1,df2,df3,df4, output_file):
    """
    将包含 lat, lng, radius 的 DataFrame 转换为指定的 JSON 格式并保存

    Args:
        df: 包含  lat, lng, radius 列的 DataFrame
        output_file: 输出的 JSON 文件路径
        unit_string: unit 字段的值
    """
    # 1. 提取全局属性 (假设整个 DataFrame 的 code 和 area 是相同的)
    # 如果有多行，取第一行的值
    # code_value = int(df.iloc[0]['code']) if 'code' in df.columns else 0
    # area_value = int(df.iloc[0]['area']) if 'area' in df.columns else 0

    # 定义文件夹名称和文件名称
    folder_name = 're_jsons'
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"文件夹 '{folder_name}' 已创建。")
    else:
        print(f"文件夹 '{folder_name}' 已存在。")


    # 2026-3-30 新增功能
    plot_edge_balls = find_max_diff(df2)
    # print('plot_edge_balls -tail', plot_edge_balls.tail())
    # plotf(plot_edge_balls)

    plot_edge_ballslist = []
    for _, row in plot_edge_balls.iterrows():
        point = {
            "lat": float(row['lat']),
            "lng": float(row['lng']),
            "radius": float(row['radius'])
        }
        plot_edge_ballslist.append(point)
    # plot_edge_ballslist.pop()
    del plot_edge_balls

    # 2. 构建 points 列表
    points_list0 = []
    points_list1 = []
    points_list2 = []
    points_list3 = []
    points_list4 = []

    for _, row in df0.iterrows():
        point = {
            "lat": float(row['lat']),
            "lng": float(row['lng']),
            "radius": float(row['radius'])
        }
        points_list0.append(point)

    for _, row in df1.iterrows():
        point = {
            "lat": float(row['lat']),
            "lng": float(row['lng']),
            "radius": float(row['radius'])
        }
        points_list1.append(point)

    for _, row in df2.iterrows():
        point = {
            "lat": float(row['lat']),
            "lng": float(row['lng']),
            "radius": float(row['radius'])
        }
        points_list2.append(point)

    for _, row in df3.iterrows():
        point = {
            "lat": float(row['lat']),
            "lng": float(row['lng']),
            "radius": float(row['radius'])
        }
        points_list3.append(point)

    for _, row in df4.iterrows():
        point = {
            "lat": float(row['lat']),
            "lng": float(row['lng']),
            "radius": float(row['radius'])
        }
        points_list4.append(point)

    # 3. 构建最终的 JSON 结构
    result_dict = {
        "code": code,
        "area": area,
        "plot_edge_balls": plot_edge_ballslist,
        "unit": '平方亩',
        "points": [
            {
                'step': 0,
                'list': points_list0,
            },
            {
                'step': 1,
                'list': points_list1,
            },
            {
                'step': 2,
                'list': points_list2,
            },
            {
                'step': 3,
                'list': points_list4,
            }

        ]

    }
    # print(result_dict)
    del code, area, plot_edge_ballslist, points_list0, points_list1, points_list2, points_list3, points_list4
    # 4. 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print(f"JSON 文件已保存为: {output_file}")
    del output_file,
    return result_dict

def main_area_computing(df, visual, plot_type, file_name):
    np.random.seed(0)

    plot_title = file_name
    del file_name
    print(plot_title)
    # ------------------------------

    df,lens = json2csv_main(df, visual, plot_type, plot_title)

    # print(df)
    # print(plot_title)
    if isinstance(df, pd.DataFrame):
        numpy_array = df.values
        del df
        X = numpy_array
        del numpy_array

        start_time = datetime.datetime.now()
        areas = divide_ball_GBC_y(X, 5, visual, plot_title)
        end_time = datetime.datetime.now()

        print('json2csv 花费时间：', end_time - start_time)
        del end_time, start_time
        return areas, lens
    else:
        return 0

    # --------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    # 测试调试主函数   需要修改


def extract_coordinates_to_df(json_file_path):
    """
    读取包含轨迹数据的JSON文件，提取经纬度并返回DataFrame
    """
    # 1. 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 提取核心数据列表
    # 假设数据结构是固定的，核心数据在 'data' 键下
    trajectory_list = data['data']
    date = data['date']

    # 3. 创建DataFrame并提取指定列
    # 先创建包含所有信息的DataFrame，再筛选需要的列
    df = pd.DataFrame(trajectory_list)[['lat', 'lng']]

    # 4. 数据类型转换（可选）
    # 确保经纬度是浮点数类型，方便后续计算
    df['lat'] = df['lat'].astype(float)
    df['lng'] = df['lng'].astype(float)

    return df,date


def plot_c(ax,df0, df, lat_max, lat_min, lng_max, lng_min):
    ax.clear()

    # 多个圆的参数
    ax.plot(df0.lat, df0.lng, alpha=0.7)

    # 绘制每个圆形
    for i in range(len(df)):
        # print((df.iloc[i, 0], df.iloc[i, 1]), df.iloc[i, 2])
        circle = patches.Circle((df.iloc[i, 0], df.iloc[i, 1]), df.iloc[i, 2],
                                fill=True, alpha=0.7,
                                facecolor='g',
                                edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)

    # 设置图形属性
    ax.set_aspect('equal')
    ax.set_xlim(lat_min - (lat_max - lat_min) * 0.2, lat_max + (lat_max - lat_min) * 0.2)
    ax.set_ylim(lng_min - (lng_max - lng_min) * 0.2, lng_max + (lng_max - lng_min) * 0.2)
    ax.grid(True, alpha=0.3)
    ax.set_title('多个圆形示例')


if __name__ == '__main__':
    rootpath = './jsons_data_34_930/'
    # files = os.listdir(rootpath)
    # print(files[:3])

    filelist1 = [
        '16170630265_2025-12-28.json',
        '16170630265_2025-12-29.json',
        '16170630270_2026-01-08.json',
        '16170630274_2026-01-04.json',
        '16170630274_2026-01-05.json'
    ]


    files = filelist1
    # files = [filelist1[0]]
    print(len(files))

    visual = False
    plot_type = 1
    results_list =[]
    for i in files:
        # print(i,type(i))
        df,date_v = extract_coordinates_to_df(rootpath+i)
        # print(df.head(),date_v)
        combined_identifier = i
        areas,lens = main_area_computing(df,visual,plot_type,combined_identifier)
        del df
        # areas = [0,5,6]
        areas_avg = np.mean([areas[2][0], areas[3][0], areas[4][0]])
        print('(areas[0][0],areas[1][0],areas[2][0],areas[3][0],areas[4][0]:',
              areas[0][0], areas[1][0], areas[2][0], areas[3][0], areas[4][0])
        results_list.append([areas[2][0], areas[3][0], areas[4][0]])
        print('avg area:', areas_avg)

        print('(areas[0][1],areas[1][1],areas[2][1],areas[3][1],areas[4][1]:',
              len(areas[0][1]), len(areas[1][1]), len(areas[2][1]), len(areas[3][1]), len(areas[4][1]))

        # 调用函数保存
        df0 = pd.DataFrame(areas[0][1], columns=['lat', 'lng', 'radius'])
        # print(df0.head())

        df1 = pd.DataFrame(areas[1][1], columns=['lat', 'lng', 'radius'])
        # print(df1.head())

        df2 = pd.DataFrame(areas[2][1], columns=['lat', 'lng', 'radius'])
        # print(df2.head())

        df3 = pd.DataFrame(areas[3][1], columns=['lat', 'lng', 'radius'])
        # print(df3.head())

        df4 = pd.DataFrame(areas[4][1], columns=['lat', 'lng', 'radius'])
        # print(df4.head())
        del areas, lens
        if areas_avg is not None:
            code = 200
        else:
            code = 999

        response_data = dataframe_to_json(code, areas_avg, df0, df1, df2, df3, df4, './re_jsons/' + combined_identifier)
        del df0, df1, df2, df3, df4, combined_identifier
        print(response_data)

        plot_edge_balls_df = pd.DataFrame(response_data['plot_edge_balls'])
        print('plot_edge_balls_df', plot_edge_balls_df.head(), len(plot_edge_balls_df))
        # 是否显示每一步的粒球显示
        v = False  # True or False
        if v:
            # 提取并保存每个step的数据
            for step_idx, step_data in enumerate(response_data['points']):
                if step_idx == 0:
                    step0_df = pd.DataFrame(step_data['list'])
                    print(len(step0_df))
                if step_idx == 1:
                    step1_df = pd.DataFrame(step_data['list'])
                    print(len(step1_df))
                if step_idx == 2:
                    step2_df = pd.DataFrame(step_data['list'])
                    print(len(step2_df))
                if step_idx == 3:
                    step3_df = pd.DataFrame(step_data['list'])
                    print(len(step3_df))

            from matplotlib.animation import FuncAnimation

            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(bottom=0.2)  # 调整底部留白

            df0 = json2csv(rootpath + i)
            df_points = plot_edge_balls_df
            # print(df_points)

            lat_max = df_points['lat'].max()
            lat_min = df_points['lat'].min()

            lng_max = df_points['lng'].max()
            lng_min = df_points['lng'].min()

            steps = [step0_df, step1_df, step2_df, step3_df]
            # 创建动画
            ani = FuncAnimation(
                fig,
                func=lambda frame: plot_c(ax, df0, steps[frame], lat_max, lat_min, lng_max, lng_min),
                frames=len(steps),
                interval=1000,  # 每帧间隔时间（毫秒）
                repeat=True
            )

            plt.show()
    print('results_list')
    for i in results_list:
        print(i[0], "\t", i[1], "\t", i[2])
    print('avg_list')
    for i in results_list:
        print(np.average([i[0], i[1], i[2]]))

