# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,flash-attn,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ERROR=false


if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --nvdiffrec) NVDIFFREC=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --flash-attn            Install flash-attention"
    echo "  --cumesh                Install cumesh"
    echo "  --o-voxel               Install o-voxel"
    echo "  --flexgemm              Install flexgemm"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --nvdiffrec             Install nvdiffrec"
    return
fi

# Get system information
echo "[SETUP] [SYSTEM INFO] START: Detect platform"
WORKDIR=$(pwd)
if command -v nvidia-smi > /dev/null; then
    PLATFORM="cuda"
elif command -v rocminfo > /dev/null; then
    PLATFORM="hip"
else
    echo "Error: No supported GPU found"
    exit 1
fi
echo "[SETUP] [SYSTEM INFO] END: Detected platform=$PLATFORM"

if [ "$NEW_ENV" = true ] ; then
    echo "[SETUP] [NEW-ENV] START: Create new pyenv environment"
    # use pyenv to create a virtual environment
    pyenv virtualenv 3.11.0 trellis2
    touch .python-version
    echo "trellis2" > .python-version
    pyenv activate trellis2

    pip install --upgrade pip setuptools wheel;

    if [ "$PLATFORM" = "cuda" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    elif [ "$PLATFORM" = "hip" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    fi
    echo "[SETUP] [NEW-ENV] END: Environment created"
fi

if [ "$BASIC" = true ] ; then
    echo "[SETUP] [BASIC] START: Install basic dependencies"
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard modelscope
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    sudo apt install -y libjpeg-dev
    pip install pillow-simd
    pip install kornia timm
    pip install psutil
    echo "[SETUP] [BASIC] END: Basic dependencies installed"
fi

if [ "$NVDIFFRAST" = true ] ; then
    echo "[SETUP] [NVDIFFRAST] START: Install nvdiffrast"
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        pip install /tmp/extensions/nvdiffrast --no-build-isolation
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
    echo "[SETUP] [NVDIFFRAST] END: nvdiffrast installed"
fi

if [ "$NVDIFFREC" = true ] ; then
    echo "[SETUP] [NVDIFFREC] START: Install nvdiffrec"
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
        pip install /tmp/extensions/nvdiffrec --no-build-isolation
    else
        echo "[NVDIFFREC] Unsupported platform: $PLATFORM"
    fi
    echo "[SETUP] [NVDIFFREC] END: nvdiffrec installed"
fi

if [ "$CUMESH" = true ] ; then
    echo "[SETUP] [CUMESH] START: Install CuMesh"
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
    pip install /tmp/extensions/CuMesh --no-build-isolation
    echo "[SETUP] [CUMESH] END: CuMesh installed"
fi

if [ "$OVOXEL" = true ] ; then
    echo "[SETUP] [O-VOXEL] START: Install o-voxel"
    mkdir -p /tmp/extensions
    cp -r o-voxel /tmp/extensions/o-voxel
    pip install /tmp/extensions/o-voxel --no-build-isolation
    echo "[SETUP] [O-VOXEL] END: o-voxel installed"
fi

if [ "$FLASHATTN" = true ] ; then
    echo "[SETUP] [FLASH-ATTN] START: Install flash-attention"
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install flash-attn==2.7.4.post1 --no-build-isolation
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        GPU_ARCHS=gfx942 python setup.py install #MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
    echo "[SETUP] [FLASH-ATTN] END: flash-attention installed"
fi

if [ "$FLEXGEMM" = true ] ; then
    echo "[SETUP] [FLEXGEMM] START: Install FlexGEMM"
    mkdir -p /tmp/extensions
    git clone --branch v1.0.0 https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
    pip install /tmp/extensions/FlexGEMM --no-build-isolation
    echo "[SETUP] [FLEXGEMM] END: FlexGEMM installed"
fi

echo "[SETUP] All requested steps completed"
