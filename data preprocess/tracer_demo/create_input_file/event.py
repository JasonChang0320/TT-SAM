import pandas as pd

event = pd.read_csv("../../events_traces_catalog/2009_2019_ok_events.csv")

event["longitude"]=event["lon"]+(event["lon_minute"]/60)
event["latitude"]=event["lat"]+(event["lat_minute"]/60)

file_path = "event_input.evt"

# KNM003 station location out of velocity model range
drop_EQ_ID = 29363
event = event[(event["EQ_ID"] != drop_EQ_ID)].reset_index()

with open(file_path, "w") as file:
    for i in range(len(event)):
        file.write(
            f"{round(event.longitude[i],3)}  {round(event.latitude[i],3)}  {event.depth[i]}\n"
        )