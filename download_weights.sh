#! /bin/bash
if [ ! -d "weights" ]; then
  mkdir weights
fi

echo "Downloading SuperPoint/LightGlue weights..."
wget -O weights/superpoint_lightglue.pth https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth

echo "Downloading Deep Homography ViT..."
wget -O weights/deep_homography_vit.pth https://huggingface.co/hudsonmsb/deep_homography_vit/resolve/main/deep_homography_vit.pth

echo "Downloading DDI-VO weights..."
wget -O weights/ddi_vo.tar https://huggingface.co/hudsonmsb/DDI-VO/resolve/main/ddi_vo.tar

echo
echo "All weights downloaded successfully!"