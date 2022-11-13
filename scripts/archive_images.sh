#!/usr/bin/env bash


# Help function.
function print_help {
  script_filename=$(basename $0)

  echo "Usage $script_filename [OPTION...]"
  echo
  echo "-d, --dir         directory path where the images are stored"
  echo "-o, --out         directory path to store image archives in"
  echo "-h, --help        print this help list"
}

# Where are the images stored?
IMG_STORAGE_DIR=""
ARCHIVE_DIR=""

# Extract options from the command line.
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"

  case $key in
    -h|--help)
      print_help
      exit 0
      ;;
    -d|--dir)
      IMG_STORAGE_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--out)
      ARCHIVE_DIR="$2"
      shift
      shift
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

# Verify required args have been passed in.
if [[ $IMG_STORAGE_DIR = "" ]]; then
  echo "Image storage path is required."
  print_help
  exit 1
elif [[ $ARCHIVE_DIR = "" ]]; then
  echo "Archival storage path is required."
  print_help
  exit 1
fi

# Ensure archive path exists.
mkdir -p $ARCHIVE_DIR
echo "Storing archives in $ARCHIVE_DIR."

# Extract created images in the past. (not today)
ARCHIVE_FILENAME=$(date +"%b-%d-%Y-%s")
echo "Creating a tar for the images in $ARCHIVE_DIR/$ARCHIVE_FILENAME.tar..."

# List all of the files in the given storage director that were not created
# today, archiving them into the given archival path.
cd $IMG_STORAGE_DIR
find . -not -name "*$(date +"%Y-%m-%d")*" -type f | \
  xargs -i sh -c "tar rvf \"$ARCHIVE_DIR/$ARCHIVE_FILENAME.tar\" '{}'; rm '{}'"
echo "Done."

# Compress the created tarball.
echo "Compressing the tarball..."
gzip -9 $ARCHIVE_DIR/$ARCHIVE_FILENAME.tar
echo "Done."

