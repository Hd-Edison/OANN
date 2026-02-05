import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_single_MVM(index1=1, index2=1):
    # 创建图形和坐标轴
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # 设置坐标轴范围
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # 绘制耦合器方块（中心位置）
    coupler = patches.Rectangle((4, 2), 2, 2, linewidth=2, edgecolor='black', facecolor='lightblue')
    ax.add_patch(coupler)

    # 绘制左侧端口（输入端口）
    ax.plot([1, 4], [3.5, 3.5], 'k-', linewidth=2)  # 端口1
    ax.plot([1, 4], [2.5, 2.5], 'k-', linewidth=2)  # 端口2

    # 绘制右侧端口（输出端口）
    ax.plot([6, 9], [3.5, 3.5], 'k-', linewidth=2)  # 端口3
    ax.plot([6, 9], [2.5, 2.5], 'k-', linewidth=2)  # 端口4

    # 添加端口标签（移至线段上方）
    ax.text(2.5, 3.8, f'In_{index1}{index2}[1]', fontsize=12, ha='center', va='center')
    ax.text(2.5, 2.8, f'In_{index1}{index2}[0]', fontsize=12, ha='center', va='center')
    ax.text(7.5, 3.8, f'Out_{index1}{index2}[1]', fontsize=12, ha='center', va='center')
    ax.text(7.5, 2.8, f'Out_{index1}{index2}[1]', fontsize=12, ha='center', va='center')

    # 添加耦合器标签
    ax.text(5, 3, f'R_{index1}{index2}', fontsize=14, ha='center', va='center', weight='bold')

    # 设置图形属性
    ax.set_aspect('equal')
    ax.axis('off')  # 隐藏坐标轴

    # 显示图形
    plt.tight_layout()
    plt.savefig(f"RMZI_{index1}{index2}", dpi=300, bbox_inches='tight')

    plt.close()


def plot_MVM(n):
    # 计算图形尺寸（需要考虑最大偏移）
    fig_width = (n + n - 1) * 4 + 2
    fig_height = n * 3 + 2

    # 创建图形和坐标轴
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # 设置坐标轴范围
    ax.set_xlim(0, (n + n - 1) * 4 + 2)
    ax.set_ylim(0, n * 3 + 2)

    # 绘制nxn个RMZI
    for row in range(1, n + 1):
        for col in range(1, n + 1):
            # 计算每个RMZI的位置，第n行向右移n-1格，第n列向上移(n-1)*0.5格
            x_offset = (col - 1 + row - 1) * 4 + 1
            y_offset = (n - row) * 0.5 + 1 + (col - 1) * 0.5  # 每列向上偏移

            # 绘制耦合器方块
            coupler = patches.Rectangle((x_offset + 1.5, y_offset + 0.5), 1, 1,
                                        linewidth=2, edgecolor='black', facecolor='lightblue')
            ax.add_patch(coupler)

            # 绘制左侧端口
            ax.plot([x_offset, x_offset + 1.5], [y_offset + 1.25, y_offset + 1.25], 'k-', linewidth=2)
            ax.plot([x_offset, x_offset + 1.5], [y_offset + 0.75, y_offset + 0.75], 'k-', linewidth=2)

            # 绘制右侧端口
            ax.plot([x_offset + 2.5, x_offset + 4], [y_offset + 1.25, y_offset + 1.25], 'k-', linewidth=2)
            ax.plot([x_offset + 2.5, x_offset + 4], [y_offset + 0.75, y_offset + 0.75], 'k-', linewidth=2)

            # 添加端口标签
            ax.text(x_offset + 0.75, y_offset + 1.4, f'In_{row}{col}[0]', fontsize=8, ha='center', va='center')
            ax.text(x_offset + 0.75, y_offset + 0.9, f'In_{row}{col}[1]', fontsize=8, ha='center', va='center')
            ax.text(x_offset + 3.25, y_offset + 1.4, f'Out_{row}{col}[0]', fontsize=8, ha='center', va='center')
            ax.text(x_offset + 3.25, y_offset + 0.9, f'Out_{row}{col}[1]', fontsize=8, ha='center', va='center')

            # 添加RMZI标签
            ax.text(x_offset + 2, y_offset + 1, f'R_{row}{col}', fontsize=10,
                    ha='center', va='center', weight='bold')


    # 设置图形属性
    ax.set_aspect('equal')
    ax.axis('off')

    # 保存图形
    plt.savefig(f"{n}x{n}_MVM", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()