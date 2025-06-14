#!/usr/bin/env bash

DATADIR="${BIMIGAN_DATA:-data}"

# Dataset download link
declare -A URL_LIST=(
    [bimigan_dataset]="https://drive.google.com/drive/folders/1MmRwlCdyXwLTZqamDkz0t3eKN6sXCObu?usp=drive_link"
)

die ()
{
    echo "${*}"
    exit 1
}

usage ()
{
    cat <<EOF
USAGE: download_dataset.sh

Download BiMIGAN dataset and organize it in the following structure:

${DATADIR}/
└── bimigan_dataset/
    ├── trainA/    # Training images from domain A
    ├── trainB/    # Training images from domain B
    ├── testA/     # Test images from domain A
    └── testB/     # Test images from domain B

Please download the dataset from:
${URL_LIST["bimigan_dataset"]}

After downloading, please extract and organize the dataset according to the structure above.
EOF

    if [[ $# -gt 0 ]]
    then
        die "${*}"
    else
        exit 0
    fi
}

dataset="${1}"

case "${dataset}" in
    help|-h|--help)
        usage
        ;;
    *)
        usage
esac

