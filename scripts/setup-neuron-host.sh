#!/usr/bin/env bash
# AWS's AL2023 Neuron installation, without upgrading the OS or rebooting.
set -euo pipefail

if [[ ${1:-} != --install ]]; then
    echo 'Check: lspci -nn | grep -i neuron; /opt/aws/neuron/bin/neuron-ls'
    echo 'Install: just setup-neuron-host --install (requires sudo)'
    exit 0
fi

source /etc/os-release
if [[ "$ID" != amzn || "$VERSION_ID" != 2023 ]]; then
    echo 'This installer supports Amazon Linux 2023; use the AWS Neuron setup guide for other OSes.' >&2
    exit 1
fi

repo_file=$(mktemp)
trap 'rm -f "$repo_file"' EXIT
cat > "$repo_file" <<'EOF'
[neuron]
name=AWS Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
gpgcheck=1
gpgkey=https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
metadata_expire=3600
EOF
if [[ -f /etc/yum.repos.d/neuron.repo ]]; then
    echo 'Using the existing Neuron repository configuration.'
else
    sudo install -m 0644 "$repo_file" /etc/yum.repos.d/neuron.repo
fi
sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
sudo dnf install -y "kernel-devel-uname-r = $(uname -r)" gcc gcc-c++ make \
    python3.11 python3.11-devel python3.12 python3.12-devel libxcrypt-compat openssl-devel \
    pkgconf-pkg-config protobuf-compiler rust cargo rustfmt clang tmux
sudo dnf install -y aws-neuronx-dkms aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools
sudo modprobe neuron
/opt/aws/neuron/bin/neuron-ls
rpm -q aws-neuronx-dkms aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools
