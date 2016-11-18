if [ $# -lt 2 ]
then
    echo "Not enough arguments supplied. Pass source and destination directories"
    exit
fi

# srcdir is the original data
srcdir=$1
# xmldir is tmp directory for adding <root> tags so it can be parsed
xmldir=xmldir
# txtdir is final text
txtdir=$2

for doc in `(ls $srcdir)`; do
    echo $doc
    full_docname=$srcdir/$doc
    cat oroot.txt $full_docname croot.txt > $xmldir/${doc}.xml
    python preprocess.py $xmldir/${doc}.xml > $txtdir/${doc}.txt
    rm $xmldir/${doc}.xml
done
