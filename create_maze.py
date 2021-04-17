#!/usr/bin/python3
'''
演示Recursive Bactracker迷宫生成方法
一个迷宫地图表示为类似于
maze = [
    [*,S,1,1],
    [*,*,*,1],
    [*,1,1,1],
    [*,T,*,*]
]
的形式。*表示墙，1表示通路。
S表示生成起点，T表示生成终点
'''
import random

class Maze:
    def __init__(self, sz: tuple, start=(0, 0)):
        '''
        sz: 表示迷宫大小的二维tuple。例如(20, 20)
        start: 表示生成起点的二维tuple
        '''
        assert len(sz) == 2
        assert len(start) == 2
        assert 0 <= start[0] < sz[0] and 0 <= start[1] < sz[1]
        self.new_size = sz
        self.size = sz[0] - 2, sz[1] - 2
        self.start = start
        self.end = (0, 0)
        rows, cols = self.size
        # generate a 2d array of size (rows, cols)
        self.maze = [[0 for j in range(cols)] for i in range(rows)]

        random.seed(22)

    def _is_next_cand(self, pos, visited):
        rows, cols = self.size
        if pos[0] < 0 or pos[0] >= rows:
            return False
        if pos[1] < 0 or pos[1] >= cols:
            return False
        return not visited[pos[0]][pos[1]]

    def __repr__(self):
        return self.maze

    def show(self):
        for line in self.maze:
            for c in line:
                print(f"{c} ", end='')
            print()
        print('start:', (self.start[0] + 1, self.start[1] + 1))
        print('end:', (self.end[0] + 1, self.end[1] + 1))

    def make_path(self):
        '''
        生成迷宫的路径
        '''
        # 维护回溯路径用的栈 bt: backtrack
        bt = []
        # 维护访问状态的表
        rows, cols = self.size
        visited = [[False for j in range(cols)] for i in range(rows)]

        cur_pos = self.start
        bt.append(cur_pos)
        visited[cur_pos[0]][cur_pos[1]] = True
        while (len(bt) > 0):
            # cur_pos = bt.pop()

            # If the current cell has any neighbours which have not been visited
            # listing the unvisted
            next_pos_cands = []
            if self._is_next_cand((cur_pos[0] - 2, cur_pos[1]), visited):
                next_pos_cands.append((cur_pos[0] - 2, cur_pos[1]))
            if self._is_next_cand((cur_pos[0] + 2, cur_pos[1]), visited):
                next_pos_cands.append((cur_pos[0] + 2, cur_pos[1]))
            if self._is_next_cand((cur_pos[0], cur_pos[1] - 2), visited):
                next_pos_cands.append((cur_pos[0], cur_pos[1] - 2))
            if self._is_next_cand((cur_pos[0], cur_pos[1] + 2), visited):
                next_pos_cands.append((cur_pos[0], cur_pos[1] + 2))

            # pdb.set_trace()

            if len(next_pos_cands) > 0:
                next_pos = random.choice(next_pos_cands)
                bt.append(next_pos)
                visited[next_pos[0]][next_pos[1]] = True
                # 打通墙壁
                y_from = cur_pos[0]
                y_to = next_pos[0]
                if y_from > y_to:
                    y_from = next_pos[0]
                    y_to = cur_pos[0]
                x_from = cur_pos[1]
                x_to = next_pos[1]
                if x_from > x_to:
                    x_from = next_pos[1]
                    x_to = cur_pos[1]
                for i in range(y_from, y_to + 1):
                    for j in range(x_from, x_to + 1):
                        self.maze[i][j] = 1
                cur_pos = next_pos
            else:
                # backtrace now
                cur_pos = bt.pop()

            # print("---------")
            # self.show()
            # pdb.set_trace()

        # 标记终点
        self.maze[next_pos[0]][next_pos[1]] = 1
        self.end = next_pos
        # 标记起点
        self.maze[self.start[0]][self.start[1]] = 1

    def improve(self):
        new_maze = [[0 for _ in range(self.new_size[1])] for _ in range(self.new_size[0])]
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                new_maze[i + 1][j + 1] = self.maze[i][j]
        self.maze = new_maze

def main():
    m = Maze((15, 15))
    m.make_path()
    m.improve()
    m.show()
    print(m.__repr__())

if __name__ == "__main__":
    main()