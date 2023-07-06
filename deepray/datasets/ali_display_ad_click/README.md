ali_display_ad_click is a dataset of click rate prediction about display Ad, 
which is displayed on the website of Taobao. The dataset is offered by the company of Alibaba. You can get the raw dataset from: https://tianchi.aliyun.com/dataset/56

The processing scripts colletced from https://github.com/DeepRec-AI/HybridBackend/tree/08baaa5ef7d27509f32a698fd55d55f5be988297/docs/tutorial/ranking/taobao/data



```
python prep_1_backbone.py \
    --raw_sample_fname /workspaces/dataset/ali_display_ad_click/raw_sample.csv \
    --user_profile_fname /workspaces/dataset/ali_display_ad_click/user_profile.csv \
    --ad_feature_fname /workspaces/dataset/ali_display_ad_click/ad_feature.csv

python prep_2_bahavior.py \
    --fname /workspaces/dataset/ali_display_ad_click/behavior_log.csv \
    --pv-output-fname /workspaces/dataset/ali_display_ad_click/pv_log.parquet \
    --cart-output-fname /workspaces/dataset/ali_display_ad_click/cart_log.parquet \
    --fav-output-fname /workspaces/dataset/ali_display_ad_click/fav_log.parquet \
    --buy-output-fname /workspaces/dataset/ali_display_ad_click/buy_log.parquet

python prep_3_merge.py \
   --pv-log-fname /workspaces/dataset/ali_display_ad_click/pv_log.parquet \
   --cart-log-fname /workspaces/dataset/ali_display_ad_click/cart_log.parquet \
   --fav-log-fname /workspaces/dataset/ali_display_ad_click/fav_log.parquet \
   --buy-log-fname /workspaces/dataset/ali_display_ad_click/buy_log.parquet \
   --fname backbone_day_0.parquet \
   --output_fname backbone_day_0_merge.parquet

```