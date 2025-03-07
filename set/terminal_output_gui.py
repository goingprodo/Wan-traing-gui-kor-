#!/usr/bin/env python3
import os
import gradio as gr
import subprocess
import toml
import glob
from datetime import datetime
import time
import shutil
import threading
import psutil
import signal

# 기본 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 위치 (set 폴더)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # 상위 폴더 (diffusion-pipe 루트)

# 상대 경로로 지정
if os.path.basename(ROOT_DIR) != 'diffusion-pipe':
    print(f"경고: 현재 디렉토리가 diffusion-pipe가 아닙니다: {ROOT_DIR}")
    print("스크립트가 diffusion-pipe/set/ 디렉토리에 있는지 확인하세요.")
    # 그래도 상위 디렉토리를 사용
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
SET_DIR = os.path.join(ROOT_DIR, "set")

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SET_DIR, exist_ok=True)

print(f"스크립트 위치: {SCRIPT_DIR}")
print(f"루트 디렉토리: {ROOT_DIR}")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print(f"설정 디렉토리: {SET_DIR}")

# 설정 옵션
DTYPES = ["bfloat16", "float16", "float32"]
TRANSFORMER_DTYPES = ["bfloat16", "float16", "float32", "float8"]
TIMESTEP_METHODS = ["logit_normal", "uniform"]

def create_dataset_config(
    dataset_dirs, min_resize_res, max_resize_res, batch_aspect_ratio,
    frame_buckets, caption_extension, image_extensions, num_repeats
):
    """데이터셋 설정 파일 생성"""
    # 프레임 버킷 변환
    frame_buckets_list = [int(x.strip()) for x in frame_buckets.split(",") if x.strip()]
    
    dataset_config = {
        "resolutions": [min_resize_res],
        "enable_ar_bucket": batch_aspect_ratio,
        "min_ar": 0.5,
        "max_ar": 2.0,
        "num_ar_buckets": 7,
        "frame_buckets": frame_buckets_list
    }
    
    # 디렉토리 목록 생성
    directories = []
    for dataset_dir in dataset_dirs.split(","):
        dataset_dir = dataset_dir.strip()
        if not dataset_dir:
            continue
            
        directory = {
            "path": dataset_dir,
            "num_repeats": num_repeats
        }
            
        directories.append(directory)
    
    # TOML 형식에 맞게 directory 키를 추가
    dataset_config["directory"] = directories
    
    return dataset_config

def generate_config(
    model_path, data_path, output_path, 
    batch_size, epochs, save_epochs, pipeline_stages,
    gradient_accumulation_steps, learning_rate, use_lora, lora_rank, 
    dtype, transformer_dtype, timestep_method
):
    """완(Wan) 모델 설정 파일 생성"""
    config = {
        # 출력 경로 설정
        "output_dir": output_path,
        
        # 데이터셋 설정
        "dataset": data_path,
        
        # 훈련 설정
        "epochs": epochs,
        "micro_batch_size_per_gpu": batch_size,
        "pipeline_stages": pipeline_stages,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "warmup_steps": 100,
        
        # 평가 설정
        "eval_every_n_epochs": 1,
        "eval_before_first_step": True,
        "eval_micro_batch_size_per_gpu": batch_size,
        "eval_gradient_accumulation_steps": 1,
        
        # 기타 설정
        "save_every_n_epochs": save_epochs,
        "checkpoint_every_n_minutes": 120,
        "activation_checkpointing": True,
        "partition_method": "parameters",
        "save_dtype": dtype,
        "caching_batch_size": 1,
        "steps_per_print": 1,
        "video_clip_mode": "single_middle",
        
        # 모델 설정
        "model": {
            "type": "wan",
            "ckpt_path": model_path,
            "dtype": dtype,
            "timestep_sample_method": timestep_method
        },
        
        # 옵티마이저 설정
        "optimizer": {
            "type": "adamw_optimi",
            "lr": learning_rate,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "eps": 1e-8
        }
    }
    
    # TensorBoard 로그 디렉토리 설정 (train.py 수정 없이 충돌 방지)
    # 실제 구현에서는 train.py를 수정하는 것이 더 좋을 수 있습니다
    config["tensorboard_dir"] = os.path.join(output_path, "tensorboard")
    
    # transformer_dtype 설정
    if transformer_dtype and transformer_dtype != dtype:
        config["model"]["transformer_dtype"] = transformer_dtype
        
    # LoRA 사용 설정
    if use_lora:
        config["adapter"] = {
            "type": "lora",
            "rank": lora_rank,
            "dtype": dtype
        }
    
    return config

def save_config(config, path):
    """TOML 설정 파일 저장"""
    with open(path, "w") as f:
        toml.dump(config, f)
    return path

def run_training(
    model_path, dataset_dirs, output_dir,
    batch_size, epochs, save_epochs, pipeline_stages, num_gpus,
    gradient_accumulation_steps, learning_rate, use_lora, lora_rank, 
    dtype, transformer_dtype, timestep_method, 
    regenerate_cache, resume_checkpoint,
    min_resize_res, max_resize_res, frame_buckets, 
    caption_extension, image_extensions,
    batch_aspect_ratio, num_repeats
):
    """훈련 설정 생성 및 터미널에 직접 실행"""
    
    # 현재 시간으로 고유한 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # TensorBoard 디렉토리 생성 (충돌 방지)
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    
    # 데이터셋 설정 생성 및 저장
    dataset_config_path = os.path.join(run_dir, "dataset.toml")
    dataset_config = create_dataset_config(
        dataset_dirs, min_resize_res, max_resize_res, batch_aspect_ratio,
        frame_buckets, caption_extension, image_extensions, num_repeats
    )
    save_config(dataset_config, dataset_config_path)
    
    # 훈련 설정 생성 및 저장
    config_path = os.path.join(run_dir, "my_wan_config.toml")
    config = generate_config(
        model_path, dataset_config_path, run_dir,
        batch_size, epochs, save_epochs, pipeline_stages,
        gradient_accumulation_steps, learning_rate, use_lora, lora_rank, 
        dtype, transformer_dtype, timestep_method
    )
    save_config(config, config_path)
    
    # 출력 디렉토리 이름만 추출 (run_YYYYMMDD_HH-MM-SS)
    output_dir_name = os.path.basename(run_dir)
    
    # 타임스탬프 폴더를 set 디렉토리 안에 복사
    set_output_dir = os.path.join(SET_DIR, output_dir_name)
    os.makedirs(set_output_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(set_output_dir, "my_wan_config.toml"))
    shutil.copy(dataset_config_path, os.path.join(set_output_dir, "dataset.toml"))
    
    # 실행 스크립트 저장 (런 디렉토리에)
    cmd_file_path = os.path.join(run_dir, "run_command.sh")
    with open(cmd_file_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {ROOT_DIR}\n")
        f.write("export NCCL_P2P_DISABLE=1\n")
        f.write("export NCCL_IB_DISABLE=1\n")
        f.write(f"deepspeed --num_gpus={num_gpus} train.py --deepspeed --config=set/{output_dir_name}/my_wan_config.toml")
        if regenerate_cache:
            f.write(" --regenerate_cache")
        if resume_checkpoint:
            f.write(" --resume_from_checkpoint") 
        f.write("\n")
    
    os.chmod(cmd_file_path, 0o755)  # 실행 권한 부여
    
    # set 폴더에도 실행 스크립트 복사
    set_cmd_path = os.path.join(SET_DIR, "run_command.sh")
    shutil.copy(cmd_file_path, set_cmd_path)
    os.chmod(set_cmd_path, 0o755)  # 실행 권한 부여
    
    # 명령어 문자열 구성 (터미널에서 직접 실행)
    cmd_str = f"cd {ROOT_DIR} && export NCCL_P2P_DISABLE=1 && export NCCL_IB_DISABLE=1 && deepspeed --num_gpus={num_gpus} train.py --deepspeed --config=set/{output_dir_name}/my_wan_config.toml"
    
    if regenerate_cache:
        cmd_str += " --regenerate_cache"
    if resume_checkpoint:
        cmd_str += " --resume_from_checkpoint"
    
    # 설정 정보 출력
    result = f"""
=== 설정 완료, 훈련 시작 ===

데이터셋: {dataset_dirs}
모델: {model_path}
작업 디렉토리: {run_dir}

설정 파일:
- 훈련 설정: {os.path.join(set_output_dir, "my_wan_config.toml")}
- 데이터셋 설정: {os.path.join(set_output_dir, "dataset.toml")}
- 실행 스크립트: {set_cmd_path}

명령어를 실행합니다...
"""
    
    print("\n" + "="*80)
    print(result)
    print("="*80 + "\n")
    
    # 터미널에 직접 실행 (os.system 사용)
    exit_code = os.system(cmd_str)
    
    if exit_code != 0:
        return f"""
훈련이 완료되었거나 오류가 발생했습니다. (종료 코드: {exit_code})

작업 디렉토리: {run_dir}
설정 파일은 저장되었으므로, 나중에 다시 실행할 수 있습니다:

bash {set_cmd_path}
"""
    else:
        return f"""
훈련이 성공적으로 완료되었습니다!

작업 디렉토리: {run_dir}
설정 파일:
- 훈련 설정: {os.path.join(set_output_dir, "my_wan_config.toml")}
- 데이터셋 설정: {os.path.join(set_output_dir, "dataset.toml")}
"""

def launch_tensorboard(output_dir):
    """텐서보드 실행"""
    try:
        cmd = ["tensorboard", "--logdir", output_dir, "--port", "6006"]
        process = subprocess.Popen(cmd)
        return f"텐서보드가 시작되었습니다. http://localhost:6006/ 에서 접속 가능합니다. (PID: {process.pid})"
    except Exception as e:
        return f"텐서보드 실행 오류: {str(e)}"

def create_interface():
    with gr.Blocks(title="Wan 모델 훈련 GUI") as app:
        gr.Markdown("# Wan 모델 훈련 GUI")
        gr.Markdown("이 GUI는 설정 파일을 생성하고 훈련을 시작합니다. 훈련 로그는 터미널에 직접 표시됩니다.")
        
        with gr.Tab("기본 설정"):
            model_path = gr.Textbox(
                label="완(Wan) 모델 경로",
                value=" ",
                info="완(Wan) 모델 체크포인트 디렉토리 경로, 절대경로를 사용하세요."
            )
            
            dataset_dirs = gr.Textbox(
                label="데이터셋 디렉토리 (쉼표로 구분)",
                value=" ",
                info="훈련 데이터가 있는 디렉토리 경로, 절대경로를 사용하세요."
            )
            
            output_dir = gr.Textbox(
                label="출력 디렉토리",
                value=OUTPUT_DIR,
                info="훈련 결과가 저장될 디렉토리, 절대경로를 사용하세요."
            )
        
        with gr.Tab("훈련 설정"):
            with gr.Row():
                batch_size = gr.Number(label="배치 크기", value=1, minimum=1, step=1, info="큰 값은 더 많은 VRAM 필요")
                epochs = gr.Number(
                    label="에폭 수", 
                    value=100, 
                    minimum=1, 
                    step=1, 
                    info="훈련 기간을 모르는 경우 높은 값으로 설정 (100+ 권장)"
                )
                save_epochs = gr.Number(
                    label="저장 간격 (에폭)", 
                    value=10, 
                    minimum=1, 
                    step=1, 
                    info="작은 데이터셋에서는 높게 설정하여 저장 파일 수 감소 (10+ 권장)"
                )
            
            with gr.Row():
                pipeline_stages = gr.Number(label="파이프라인 스테이지 수", value=1, minimum=1, step=1, info="GPU 메모리가 부족하면 2 이상 설정")
                num_gpus = gr.Number(label="GPU 수", value=1, minimum=1, step=1)
                gradient_accumulation_steps = gr.Number(label="그래디언트 누적 단계", value=1, minimum=1, step=1, info="배치 사이즈를 효과적으로 늘리기 위함")
            
            with gr.Row():
                learning_rate = gr.Number(label="학습률", value=0.00001, precision=8)
                dtype = gr.Dropdown(label="데이터 타입", choices=DTYPES, value="bfloat16")
                transformer_dtype = gr.Dropdown(
                    label="트랜스포머 데이터 타입",
                    choices=[""] + TRANSFORMER_DTYPES,
                    value="float8",
                    info="float8 사용 시 VRAM 사용량 감소"
                )
            
            with gr.Row():
                use_lora = gr.Checkbox(label="LoRA 사용", value=True, info="전체 모델 대신 LoRA 어댑터만 훈련")
                lora_rank = gr.Slider(label="LoRA 랭크", value=8, minimum=1, maximum=128, step=1, info="높을수록 더 많은 파라미터 조정 가능")
            
            timestep_method = gr.Dropdown(
                label="타임스텝 샘플링 방법",
                choices=TIMESTEP_METHODS,
                value="logit_normal",
                info="logit_normal이 완 모델에 권장됨"
            )
            
            with gr.Row():
                regenerate_cache = gr.Checkbox(label="캐시 재생성", value=False, info="데이터셋 변경 시 체크")
                resume_checkpoint = gr.Checkbox(label="체크포인트에서 재개", value=False, info="중단된 훈련 재개 시 체크")
        
        with gr.Tab("데이터셋 설정"):
            with gr.Row():
                min_resize_res = gr.Number(label="최소 리사이즈 해상도", value=512, minimum=64, step=8)
                max_resize_res = gr.Number(label="최대 리사이즈 해상도", value=768, minimum=64, step=8, info="높은 해상도는 더 많은 VRAM 필요")
            
            frame_buckets = gr.Textbox(
                label="프레임 버킷 (쉼표로 구분)",
                value="1, 33",
                info="비디오 프레임 수를 그룹화하는 버킷 (첫 번째는 항상 1이어야 함)"
            )
            
            with gr.Row():
                caption_extension = gr.Textbox(label="캡션 확장자", value="txt")
                image_extensions = gr.Textbox(
                    label="이미지 확장자 (쉼표로 구분)", 
                    value="png, jpg, jpeg, webp"
                )
            
            with gr.Row():
                batch_aspect_ratio = gr.Checkbox(label="비율별 배치 그룹화", value=True, info="비슷한 비율의 이미지를 함께 처리")
                num_repeats = gr.Number(label="데이터셋 반복 횟수", value=10, minimum=1, step=1, info="1 에폭 내에서 데이터셋을 반복하는 횟수")
        
        with gr.Row():
            start_btn = gr.Button("훈련 시작", variant="primary")
            tensorboard_btn = gr.Button("텐서보드 실행")
        
        output = gr.Textbox(label="상태 출력", lines=15)
        gr.Markdown("""
### 주의사항
* 훈련 시작 버튼을 클릭하면 설정 파일이 생성되고 훈련이 시작됩니다.
* 훈련 로그는 이 GUI를 실행한 터미널 창에 직접 표시됩니다.
* GUI를 닫아도 훈련은 계속 진행됩니다.
* 훈련을 중단하려면 터미널에서 Ctrl+C를 누르세요.
* 에폭 수는 기본값 100으로 설정되어 있으며, 저장 간격은 10입니다.
        """)
        
        # 이벤트 연결
        start_btn.click(
            fn=run_training,
            inputs=[
                model_path, dataset_dirs, output_dir,
                batch_size, epochs, save_epochs, pipeline_stages, num_gpus,
                gradient_accumulation_steps, learning_rate, use_lora, lora_rank, 
                dtype, transformer_dtype, timestep_method,
                regenerate_cache, resume_checkpoint,
                min_resize_res, max_resize_res, frame_buckets,
                caption_extension, image_extensions,
                batch_aspect_ratio, num_repeats
            ],
            outputs=output
        )
        
        tensorboard_btn.click(
            fn=launch_tensorboard,
            inputs=[output_dir],
            outputs=output
        )
        
    return app

if __name__ == "__main__":
    try:
        import toml
        import gradio
        import psutil
    except ImportError as e:
        print(f"필요한 라이브러리가 설치되어 있지 않습니다: {e}")
        print("다음 명령으로 설치하세요:")
        print("pip install gradio toml psutil")
        exit(1)
    
    # GUI 실행
    app = create_interface()
    app.launch(share=False)
