import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    # assert params.input_path.endswith('.mp4')
    # parsed_name = os.path.basename(params.input_path).replace('.mp4', '').replace('sample', '').replace('rep', '')
    # sample_i, rep_i = [int(e) for e in parsed_name.split('_')]

    assert params.input_path.endswith('.mp4')
    basename = os.path.basename(params.input_path).replace('.mp4', '')

    # 尝试从文件名中智能提取 sample_i 和 rep_i
    import re
    m = re.search(r'(\d+).*?(\d+)', basename)
    if not m:
        raise ValueError(f"Cannot parse sample/rep index from filename: {basename}")
    sample_i, rep_i = int(m.group(1)), int(m.group(2))

    npy_path = os.path.join(os.path.dirname(params.input_path), 'results.npy')
    out_npy_path = params.input_path.replace('.mp4', '_smpl_params.npy')
    assert os.path.exists(npy_path)
    results_dir = params.input_path.replace('.mp4', '_obj')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                device=params.device, cuda=params.cuda)

    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)
