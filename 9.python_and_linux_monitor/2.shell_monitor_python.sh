lines=`cat ./config.txt | awk '{print $0}'`
for line in $lines
do
   datetime=`date +%Y:%m:%d:%H:%M:%S`
   data=(`echo $line | tr ':' ' '` )
   id=`netstat -anp|grep ${data[0]}|awk '{printf $7}'|cut -d/ -f1`
   len=`echo ${id} | awk '{print length($0)}'`
   if [ $len -eq 0 ]; then
      process=`${data[1]}`
      echo $datetime'：端口号：'${data[0]}'的命令：'${data[1]}'启动成功！'
   else
      echo $datetime'：端口号：'${data[0]}'的命令：'${data[1]}'无需重启！'
   fi
done

echo 'config.txt example: 5001'