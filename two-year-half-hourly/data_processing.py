import csv
from datetime import datetime, timedelta

def row_generator(path):
    with open(path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            yield row

def save_array_to_csv(arr):
    with open('two-year-half-hourly/hourly-data-processed.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(arr)
    add_timestamps()
        
def add_timestamps():
    with open('two-year-half-hourly/hourly-data-processed.csv', 'r', newline='') as infile, open('two-year-half-hourly/hourly-data-processed-timestamps.csv', 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        current_date = None
        current_time = timedelta(hours=0, minutes=30)
        
        for row in reader:
            date = row[0]
            usage = row[1]
            
            if date != current_date:
                current_date = date
                current_time = timedelta(hours=0, minutes=0)
            
            timestamp = datetime.strptime(date, '%y/%m/%d') + current_time
            time_str = timestamp.strftime('%H:%M')
            
            writer.writerow([date, time_str, usage])

            current_time += timedelta(minutes=30)
            if current_time.total_seconds() >= 24 * 60 * 60:
                current_time = timedelta(hours=0, minutes=0)

path = 'two-year-half-hourly/hourly-data-2022-2024.csv'
row_gen = row_generator(path)

energy_arr = []

add_timestamps()

while True:
    try:
        row = next(row_gen)
        if row[0] != ' ':
            date = row[0]
        energy = float(row[1]) + float(row[2])
        new_row = [date, energy]
        energy_arr.append(new_row)
    except StopIteration:
        break