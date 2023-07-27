python prep_1_backbone.py \
    --raw_sample_fname /workspaces/dataset/ali_display_ad_click/raw_sample.csv \
    --user_profile_fname /workspaces/dataset/ali_display_ad_click/user_profile.csv \
    --ad_feature_fname /workspaces/dataset/ali_display_ad_click/ad_feature.csv \
    --output-fname-template /workspaces/dataset/ali_display_ad_click/tmp/backbone_day_{}.parquet

python prep_2_bahavior.py \
    --fname /workspaces/dataset/ali_display_ad_click/behavior_log.csv \
    --pv-output-fname /workspaces/dataset/ali_display_ad_click/tmp/pv_log.parquet \
    --cart-output-fname /workspaces/dataset/ali_display_ad_click/tmp/cart_log.parquet \
    --fav-output-fname /workspaces/dataset/ali_display_ad_click/tmp/fav_log.parquet \
    --buy-output-fname /workspaces/dataset/ali_display_ad_click/tmp/buy_log.parquet

# xargs 命令的 -n 1 参数表示每次传递一个参数给 bash -c 命令，-P 3 参数表示最多并发运行3个进程。
seq 1 7 | xargs -n 1 -P 8 -I {} bash -c "
echo 'merging: {}'
python prep_3_merge.py \
  --pv-log-fname /workspaces/dataset/ali_display_ad_click/tmp/pv_log.parquet \
  --cart-log-fname /workspaces/dataset/ali_display_ad_click/tmp/cart_log.parquet \
  --fav-log-fname /workspaces/dataset/ali_display_ad_click/tmp/fav_log.parquet \
  --buy-log-fname /workspaces/dataset/ali_display_ad_click/tmp/buy_log.parquet \
  --fname /workspaces/dataset/ali_display_ad_click/tmp/backbone_day_{}.parquet \
  --output_fname /workspaces/dataset/ali_display_ad_click/tmp/merged_day_{}.parquet
if [ $? -ne 0 ]; then
  echo 'An error occurred while running the script for day {}.'
fi"

echo "sorting: $i"
python prep_4_sort.py \
    --fname-template /workspaces/dataset/ali_display_ad_click/tmp/merged_day_{}.parquet \
    --output-fname-template /workspaces/dataset/ali_display_ad_click/output/day_{}.parquet
