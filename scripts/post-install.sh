#!/bin/bash

if [[ "$1" != "" ]]; then
    NEW_PROJECT_NAME=$1
else
    read -p "Enter the new project name: " NEW_PROJECT_NAME
fi

OLD_PROJECT_NAME="PROJECT_NAME"
SEARCH_DIR=$( cd "$(dirname $(dirname "${BASH_SOURCE[0]}"))" ; pwd -P )

# Replace the value of OLD_PROJECT_NAME with $1 in every file in the project
# except for the post-install.sh script
grep $OLD_PROJECT_NAME $SEARCH_DIR -lr --exclude="*post-install.sh" | xargs sed -i '' -e 's/'$OLD_PROJECT_NAME'/'$NEW_PROJECT_NAME'/g'
