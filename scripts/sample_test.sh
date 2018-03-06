mkdir ~/repos/gazenet/data/$2/test
for file in $(ls $2 | shuf -n 100)
do
mv $file $2/test
done
