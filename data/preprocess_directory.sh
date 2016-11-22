if [ $# -lt 2 ]
then
    echo "Not enough arguments supplied. Pass destination directory and source files"
    exit
fi


txtdir=$1
echo "Destination: $txtdir"
shift
# srcdir is the original data
srcdir=`dirname $1`
echo "Directory: $srcdir"
# xmldir is tmp directory for adding <root> tags so it can be parsed
xmldir="/tmp"

for doc; do
    doc=`basename $doc`
    echo $doc
    full_docname=$srcdir/$doc
    cat oroot.txt $full_docname croot.txt > $xmldir/${doc}.xml
    python preprocess.py $xmldir/${doc}.xml > $txtdir/${doc}.txt
    rm $xmldir/${doc}.xml
done
