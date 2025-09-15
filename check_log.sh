#!/bin/bash

# Print filename if last line does NOT contain "Group-level analysis pipeline completed successfully"

for file in *.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 4 "$file" 2>/dev/null)
        if [[ "$last_line" != *"successfully"* ]]; then
            echo "$file"
        fi
    fi
done

for file in *power*.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 4 "$file" 2>/dev/null)
        if [[ "$last_line" != *"successfully"* ]]; then
            echo "$file"
        fi
    fi
done

for file in *schaefer*.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 4 "$file" 2>/dev/null)
        if [[ "$last_line" != *"successfully"* ]]; then
            echo "$file"
        fi
    fi
done