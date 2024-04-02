#!/bin/bash

# Script to fetch data from API using curl with dynamic date inputs

# Function to display help message
show_help() {
  echo "Usage: $0 <start_date> <end_date>"
  echo "Both dates should be in the 'YYYY-MM-DD' format."
  echo "Example: $0 2019-01-01 2024-01-01"
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Error: You must provide exactly two arguments."
  show_help
  exit 1
fi

# Assign the start and end date from script arguments
START_DATE=$1
END_DATE=$2

# Check if date formats are correct (basic check)
if ! [[ $START_DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || ! [[ $END_DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Error: Date format is incorrect."
  show_help
  exit 1
fi

# URL encode the query part of the URL
QUERY="SELECT%20*%20,%20ST_Y(the_geom)%20AS%20lat,%20ST_X(the_geom)%20AS%20lng%20FROM%20incidents_part1_part2%20WHERE%20dispatch_date_time%20%3E=%20'$START_DATE'%20AND%20dispatch_date_time%20%3C%20'$END_DATE'"

# Full API endpoint with the query
URL="https://phl.carto.com/api/v2/sql?filename=incidents_part1_part2&format=csv&q=$QUERY"

# Use curl to fetch the data
curl -o "data_${START_DATE}_to_${END_DATE}.csv" "$URL"
