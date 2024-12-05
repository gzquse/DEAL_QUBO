#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
usage() {
    echo "Usage: $0 -f <Dockerfile_path> -t <image_tag> [-p]"
    echo "     -p: (optional) publish image for all NERSC users"
    echo example
    echo "   ./podman-hpc-build.sh  -f Dockerfile.ubu24-xeyes -t ubu24-xeyes:p1c   -p   "

  exit 1
}

# Initialize variables
dockerfileName=""
imageName=""
use_p=false

# Parse command line arguments
while getopts ":f:t:p" opt; do
  case $opt in
    f)
      dockerfileName=$OPTARG
      ;;
    t)
      imageName=$OPTARG
      ;;
    p)
      use_p=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$dockerfileName" ] || [ -z "$imageName" ]; then
  usage
fi

# Check if the Dockerfile exists
if [ ! -f "$dockerfileName" ]; then
  echo "Error: Dockerfile '$dockerfileName' does not exist."
  exit 1
fi

if [[ "$imageName" == */* ]]; then
    echo "Error: imageName=$imageName     contains '/'"
    exit 2
fi

# check if image exist in repo already
if podman-hpc images --format "{{.Repository}}:{{.Tag}}" |grep -q $imageName ; then
    echo "Image $imageName exists, change the name"
    exit 3
fi

#  prefix image name with user name
imageName=$USER/$imageName
echo building image: $imageName ....

myProj=`sacctmgr show user $USER  withassoc format=DefaultAccount |tail -n1 | xargs`

# Display the arguments
echo "Dockerfile: $dockerfileName"
echo "Image name: $imageName"
echo "Use -p: $use_p"
echo "my NERSC project: $myProj"

# Execute the podman build command
time podman-hpc build -f "$dockerfileName" -t "$imageName"


echo 'podman-hpc images          # check image is visible'

# Execute additional commands if -p is used
if [ "$use_p" = true ]; then
    CORE_PUB="/cfs/cdirs/$myProj/$USER/podman_common/"
    echo CORE_PUB=$CORE_PUB
    podman-hpc --squash-dir "/global/$CORE_PUB" migrate "$imageName"
    chmod -R a+rx  /global/$CORE_PUB
    echo
    echo public image use example; echo
    #echo POD_PUB=/dvs_ro$CORE_PUB
    echo "export PODMANHPC_ADDITIONAL_STORES=/dvs_ro$CORE_PUB"
else
    echo private image use example; echo
    podman-hpc  migrate "$imageName"
fi

#... common
echo IMG=$imageName 
echo 'podman-hpc run -it --gpu -e DISPLAY  -v $HOME:$HOME -e HOME  $IMG  bash     # start the image'
echo

exit 0

# xeyes
