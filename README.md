# incident_insight_matrix
<b>Work in Progress</b>
Incident Insight Matrix leverages call-for-service data to predict emerging crime clusters. Through visual analysis and machine learning, it helps strategic planners allocate resources and manage public safety trends.

## GPS Ankle Monitor Program
The program collects GPS data from an ankle monitor monitoring facility, capturing the following attributes for each parolee: Parolee ID, Reference Number, GPS Fix Time (UTC), Latitude, Longitude, Dilution of Precision (DOP), Accuracy Estimate, Received Time, Speed, Satellite Count, Point Type, and PO Group.
During processing, a local datetime field is generated to convert GPS Fix Time from UTC to the agency's local timezone. Data collection is automated via Task Scheduler, and retention policies enforce a standard two-week storage period unless agency-specific policies dictate otherwise. After retention time, the file is automatically deleted by the system. This GPS data is primarily analyzed in connection with major violent incidents.

The program integrates this GPS data with the agency’s Call for Service (CFS) system, which provides geographic coordinates for each call. When a violent crime — such as a shooting, stabbing, or robbery — is reported, the system triggers the search_gps_area function. This function compares the CFS coordinates and incident time window against historical GPS data, identifying any parolee activity within a 1,000-foot radius of the event.

This capability was developed by the Real-Time Crime Center (RTCC) of Canton following a homicide investigation, where a delay in cross-referencing GPS data resulted in missed opportunities for immediate investigative leads.

The data may exceed excel's maximum number of rows allowed. If this occurs, the system will save the data into two or more separate excel files as seen below:
Number of rows total: 1,378,688
Chunk 1 saved to gps_data/gps_2025-04-22T22-17_2025-04-26T21-13_part1.xlsx
Chunk 2 saved to gps_data/gps_2025-04-22T22-17_2025-04-26T21-13_part2.xlsx

If no Excel files are present in the folder, the system automatically pulls data for the past two days, which is the maximum range supported by the API. If Excel files exist, the system identifies the most recent file, extracts the latest recorded datetime, and uses that timestamp as the new starting point for data collection.

The gps_area_search pulls from the functions file using search_gps_area(dt_start, dt_end, lat, lon, radius). If variables are not entered, it will prompt the user to fill all but the radius. By default, the radius from the coordinate given is 1,000 feet.

search_gps_area(dt_start="2025-04-26T21:20", 
    dt_end="2025-04-26T21:55", 
    lat=40.8808,
    lon=-84.5842, 
    radius=1000)

## Calls For Service Program (New World Tyler Technologies)
