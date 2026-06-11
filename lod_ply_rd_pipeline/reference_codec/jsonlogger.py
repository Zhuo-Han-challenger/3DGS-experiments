#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

from __future__ import annotations

import json
import atexit
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path
import math
import csv
import scipy.interpolate
import numpy as np
from collections import defaultdict

# 全局文件锁字典
_FILE_LOCKS = defaultdict(threading.RLock)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))

def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff

class JSONLogger:
    def __init__(self,
                 path: str,
                 default_scene: str = 'scene',
                 load_on_init: bool = True,
                 autosave: bool = True):
        self.path = Path(path)
        self.scene_name = default_scene
        self._local_lock = threading.RLock()
        self._file_lock = _FILE_LOCKS[str(self.path.resolve())]
        self._data: List[Dict[str, Dict[str, Any]]] = []
        self.item_key = "r0"
        if load_on_init and self.path.exists():
            try:
                self.load()
            except Exception:
                self._data = []
        if autosave:
            atexit.register(self.save)
    def set_path(self, path: str):
        self.path = Path(path)

    def set_scene_name(self, name: str):
        with self._local_lock:
            self.scene_name = name
    
    def set_item_key(self, key:str):
        self.item_key = key

    def load(self) -> None:
        with self._file_lock:
            if not self.path.exists():
                self._data = []
                return
            raw = json.loads(self.path.read_text(encoding='utf-8'))
            if isinstance(raw, list):
                self._data = [el for el in raw if isinstance(el, dict)]
            elif isinstance(raw, dict):
                self._data = [raw]
            else:
                self._data = []

    def save(self, path: Optional[str] = None) -> None:
        with self._file_lock:
            target = Path(path) if path else self.path
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + '.tmp')
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(target)

    def _find_scene_index(self, scene_name: str) -> Optional[int]:
        for i, el in enumerate(self._data):
            if scene_name in el:
                return i
        return None

    def log(self, field_dict: Dict[str, Any], item_key: str = None) -> None:
        with self._file_lock:
            self.load()
            if item_key is None:
                item_key = self.item_key
            idx = self._find_scene_index(self.scene_name)
            if idx is None:
                self._data.append({self.scene_name: {item_key: field_dict}})
                self.save()
                return
            scene_dict = self._data[idx][self.scene_name]
            if item_key not in scene_dict:
                scene_dict[item_key] = {}
            if isinstance(scene_dict[item_key], dict):
                scene_dict[item_key].update(field_dict)
            else:
                scene_dict[item_key] = field_dict
            self.save()

    def compute_averages(self, exclude_scenes: Optional[List[str]] = None, save: bool = True) -> Dict[str, Dict[str, float]]:
        if exclude_scenes is None:
            exclude_scenes = ['average']
        with self._file_lock:
            self.load()
            accum: Dict[str, Dict[str, List[float]]] = {}
            for scene in self._data:
                for scene_name, items in scene.items():
                    if scene_name in exclude_scenes:
                        continue
                    if not isinstance(items, dict):
                        continue
                    for item_key, item_fields in items.items():
                        if not isinstance(item_fields, dict):
                            continue
                        if item_key not in accum:
                            accum[item_key] = {}
                        for k, v in item_fields.items():
                            if _is_number(v):
                                accum[item_key].setdefault(k, []).append(float(v))
            out: Dict[str, Dict[str, float]] = {}
            for item_key, fields in accum.items():
                out[item_key] = {k: sum(vals) / len(vals) for k, vals in fields.items() if vals}
            avg_idx = self._find_scene_index('average')
            if avg_idx is None:
                self._data.append({'average': out})
            else:
                self._data[avg_idx]['average'] = out
            if save:
                self.save()
            return out

    def export_to_csv(self, csv_path: str) -> None:
        """
        Export JSON contents to CSV with columns: scene_name, item_key, followed by field keys.
        """
        rows = []
        field_names_set = set()
        for scene in self._data:
            for scene_name, items in scene.items():
                if not isinstance(items, dict):
                    continue
                for item_key, fields in items.items():
                    if not isinstance(fields, dict):
                        continue
                    row = {'scene': scene_name, 'rate': item_key}
                    for fk, fv in fields.items():
                        row[fk] = fv
                        field_names_set.add(fk)
                    rows.append(row)
        fieldnames = ['scene', 'rate'] + sorted(field_names_set)
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _load_file(path: Path) -> List[Dict[str, Dict[str, Any]]]:
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(raw, list):
            return raw
        elif isinstance(raw, dict):
            return [raw]
        else:
            return []

    @staticmethod
    def plot_files(
        file_paths: List[str],
        x_key: str,
        y_keys: List[str],
        scene_names: Optional[List[str]] = None,
        save_dir: Optional[str] = None,
        show: bool = True,
        exclude_items: Optional[List[str]] = [],
        classifier: Optional[Callable[[str], str]] = None
    ) -> None:
        if plt is None:
            raise RuntimeError('matplotlib not available')

        files = [Path(p) for p in file_paths]
        file_contents = {}
        union_scenes = set()

        # 读取所有文件内容
        for f in files:
            data = JSONLogger._load_file(f)
            file_contents[str(f.name)] = data
            for scene in data:
                for k in scene.keys():
                    union_scenes.add(k)

        if scene_names is None:
            scene_names = sorted(list(union_scenes))

        out_dir = Path(save_dir) if save_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        total_scene_points = defaultdict(list)
        names = []

        # 用于生成综合 CSV 文件的数据
        combined_data = defaultdict(list)
        combined_data_for_avg = defaultdict(list)

        for y_key in y_keys:
            total_bdrate = 0
            for scene in scene_names:
                plt.figure()
                plt.xlabel(x_key)
                plt.ylabel(y_key)
                plotted = False
                total_points = []

                for fname, data in file_contents.items():
                    # 使用字典来按类别存储数据点
                    category_points = defaultdict(list)
                    names.append(fname)

                    for entry in data:
                        if scene in entry and isinstance(entry[scene], dict):
                            item_keys_sorted = sorted(list(entry[scene].keys()))
                            for item_key, fields in sorted(entry[scene].items()):
                                if not isinstance(fields, dict):
                                    continue
                                if item_key in exclude_items:
                                    continue
                                rate_idx = item_keys_sorted.index(item_key)

                                xv = fields.get(x_key)
                                yv = fields.get(y_key)
                                if _is_number(xv) and _is_number(yv):
                                    x = float(xv)
                                    y = float(yv)

                                    category = str(scene)
                                    category_points[category].append((x, y))

                                    combined_data[(scene, fname, category)].append((x, y, y_key))
                                    combined_data_for_avg[(fname, str(rate_idx), y_key)].append((x, y, scene))

                    # 绘制每个类别
                    for category, points in category_points.items():
                        if not points:
                            continue
                        xs, ys = zip(*points)
                        plt.plot(xs, ys, marker='o', label=f"{fname}")
                        total_points.append(points)
                        plotted = True

                # # 计算 BD-Rate（如果存在至少两个文件）
                if total_points and len(total_points) == 2:
                    rate1, metric1 = zip(*total_points[0])
                    rate2, metric2 = zip(*total_points[1])
                    bd_rate = BD_RATE(rate1, metric1, rate2, metric2)
                    plt.title(f"Rate-{y_key} Curve, BD-Rate: {bd_rate:.2f}")
                    print(scene, f"bdrate: {bd_rate:2f}")
                    total_bdrate += bd_rate
                else:
                    plt.title(f"Rate-{y_key} Curve")
                total_scene_points[y_key].append(total_points)

                if plotted:
                    plt.legend()
                    if out_dir:
                        plt.savefig(out_dir / f"{scene}_{y_key}.png")
                    if show:
                        plt.show()
                else:
                    plt.close()
            print(f"avg bdrate: {(total_bdrate/len(scene_names)):2f}")
        def average_of_lists(lists):
            if not lists:
                return []

            # 检查所有列表的长度是否相同
            list_length = len(lists[0])
            if any(len(lst) != list_length for lst in lists):
                raise ValueError("所有列表的长度必须相同")

            # 计算每个位置的平均值
            averaged_list = [sum(elements) / len(lists) for elements in zip(*lists)]

            return averaged_list

        # plot average figures
        for y_key in y_keys:
            plt.figure()
            plt.xlabel(x_key)
            plt.ylabel(y_key)
            method_ax = [[] for x in file_contents.keys()]
            method_ay = [[] for x in file_contents.keys()]
            for scene in total_scene_points[y_key]:
                for ind,cat in enumerate(scene):
                    axs, ays = zip(*cat)
                    method_ax[ind].append(axs)
                    method_ay[ind].append(ays)
            for ind in range(len(file_contents.keys())):
                method_ax[ind] = average_of_lists(method_ax[ind])
                method_ay[ind] = average_of_lists(method_ay[ind])
                plt.plot(method_ax[ind], method_ay[ind], marker='o', label=f"{list(file_contents.keys())[ind]}")
            plt.title(f"Average Rate-{y_key} Curve")
            plt.legend()
            if out_dir:
                plt.savefig(out_dir / f"average_{y_key}.png")

        # 生成综合的 CSV 文件
        if out_dir:
            combined_csv_file_path = out_dir / "combined_data.csv"
            with open(combined_csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                header = [x_key, 'file_name', 'category'] + y_keys
                csvwriter.writerow(header)

                # 生成综合 CSV 文件的数据
                for (scene, fname, category), points in combined_data.items():
                    data_dict = defaultdict(list)
                    for x, y, yk in points:
                        data_dict[x].append((yk, y))

                    for x, y_values in data_dict.items():
                        row = [x, fname, category]
                        for yk in y_keys:
                            y = next((v for k, v in y_values if k == yk), None)
                            row.append(y if y is not None else '')
                        csvwriter.writerow(row)

            # 生成平均数据 CSV 文件
            averaged_csv_file_path = out_dir / "averaged_data.csv"
            with open(averaged_csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                header = ['file_name', 'rate'] + sorted(y_keys)
                csvwriter.writerow(header)
                
                avg_data = defaultdict(lambda: defaultdict(dict))
                rate_x_values = defaultdict(dict)
                
                for (fname, rate, yk), points in combined_data_for_avg.items():
                    avg_x = sum(p[0] for p in points) / len(points)
                    avg_y = sum(p[1] for p in points) / len(points)
                    rate_x_values[(fname, rate)] = avg_x
                    avg_data[(fname, rate)][yk] = avg_y
                
                for (fname, rate), yk_values in sorted(avg_data.items()):
                    row = [fname, rate]
                    for yk in sorted(y_keys):
                        row.append(yk_values.get(yk, ''))
                    csvwriter.writerow(row)


# ----------------- Example CLI usage ----------------- #
if __name__ == '__main__':
    # tiny demonstration when run as script
    import argparse
    p = argparse.ArgumentParser()
    
    p.add_argument('--exps', nargs='+',  required=False)
    p.add_argument('--figure', required=False)
    args = p.parse_args()
    jl = JSONLogger("path.json")
    exps = args.exps
    figure = args.figure
    jl.plot_files([f'./results/{exp}/{exp}.json' for exp in exps], 'total_size', ['PSNR','SSIM','LPIPS'], save_dir=f"metrics/{figure}")

    print(f"comparision has been saved in metrics/{figure}")
