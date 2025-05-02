import pandas as pd
import numpy as np
from collections import Counter
from dotenv import load_dotenv, dotenv_values
from meteostat import Hourly, Point
from datetime import datetime, timedelta
from geopy.distance import geodesic
import ollama, os, re, requests, time, h3
import pytz, urllib3, pyodbc

## === LOAD VARIABLES === ##
class GPSConfig:
    BASE_URL = None
    AUTHORIZE_ENDPOINT = None
    PORT = None
    SOURCE_ID = None
    GPSCREDENTIALS = None
    HEADERS_TEMPLATE = None
    TOKEN = None
    TOKEN_TIMESTAMP = None
    TOKEN_EXPIRY = 59 * 60  # 59 minutes
    GPS_DATA_DIR = "resources/gps_data"

## === Date & Operations Functions === ##
def today(fmt=None):
    return _format_date(datetime.now(), fmt)

def yesterday(fmt=None):
    return _format_date(today() - timedelta(days=1), fmt)

def last_week(fmt=None):
    return _format_date(today() - timedelta(days=7), fmt)

def last_month(fmt=None):
    return _format_date(today() - timedelta(days=30), fmt)

def last_quarter(fmt=None):
    return _format_date(today() - timedelta(days=90), fmt)

def last_year(fmt=None):
    return _format_date(today() - timedelta(days=365), fmt)

def _format_date(date_obj, fmt):
    """Helper function to format the date if needed."""
    return date_obj.strftime("%Y-%m-%d") if fmt == "str" else date_obj

def delete_old_files(folder_path, age_days=28):
    """
    Deletes files in the specified folder that are older than 'age_days'.
    
    :param folder_path: Path to the folder where files should be checked.
    :param age_days: Number of days old a file must be to be deleted.
    """
    current_time = time.time()
    age_seconds = age_days * 86400  # Convert days to seconds

    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)

            if file_age > age_seconds:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

def safe_latlng_to_hex(lat, lon, resolution=7):
    # Check if coordinates are within valid range
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return h3.latlng_to_cell(lat, lon, resolution)
    else:
        return None  # Or use "Invalid" if you prefer string-based

def server_connect(server, uid, pwd, db, driver="SQL SERVER"):
    """
    Establishes a connection to the SQL Server database.
    
    Args:
        server (str): The database server name.
        uid (str): The username for authentication.
        pwd (str): The password for authentication.
        db (str): The database name.
        driver (str, optional): The ODBC driver to use. Defaults to "SQL SERVER".
    
    Returns:
        pyodbc.Connection: A connection object if successful, None otherwise.
    """
    try:
        conn = pyodbc.connect(
            f"DRIVER={driver};SERVER={server};DATABASE={db};UID={uid};PWD={pwd}",
            timeout=10  # Set a timeout to avoid indefinite hanging
        )
        print("Database connection established successfully.")
        return conn
    except pyodbc.Error as e:
        print(f"Database connection failed: {e}")
        return None

def get_cliff_notes(notes: str, elaborate: bool = False, style: str = 'paragraph') -> dict:
    """ EXAMPLES BELOW """
        # Get a concise summary in paragraph form without elaboration
        # summary_paragraph = get_cliff_notes(notes_text, elaborate=False, style='paragraph')
        # print("Concise Paragraph Summary:\n", summary_paragraph['concise'])

        # # Get both a concise summary in list format and a detailed elaboration
        # summary_list = get_cliff_notes(notes_text, elaborate=True, style='listed')
        # print("\nConcise List Summary:\n", summary_list['concise'])
        # print("\nDetailed Elaboration (List):\n", summary_list['detailed'])
    
    # Determine the formatting instruction based on the chosen style
    if style.lower() == 'listed':
        format_instruction = "in a list format with bullet points"
    else:
        format_instruction = "in paragraph form"
    
    # Create the base prompt for the concise summary
    base_prompt = (
        f"Please provide a concise, very brief 'just the facts' summary of the following text. "
        f"Focus on the key facts and main points and provide a two or three sentence reply {format_instruction} that includes the outcome of the situation:\n\n{notes}"
    )
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": base_prompt}])
    concise_summary = response['message']['content']
    
    result = {'concise': concise_summary}
    
    # If elaboration is requested, create a follow-up prompt based on the style
    if elaborate:
        if style.lower() == 'listed':
            follow_up_instruction = (
                "Please elaborate further in a list format with bullet points, providing additional context and details as needed, "
                "excluding the expansion of abbreviations and including the outcome of the situation."
            )
        else:
            follow_up_instruction = (
                "Please elaborate in paragraph form and provide additional context or details as needed, "
                "excluding the expansion of abbreviations and include the outcome of the situation."
            )
        
        follow_up_prompt = f"Based on this summary:\n\n{concise_summary}\n\n{follow_up_instruction}"
        detailed_response = ollama.chat(model='mistral', messages=[{"role": "user", "content": follow_up_prompt}])
        result['detailed'] = detailed_response['message']['content']
    
    return result

def extract_names(text: str) -> list:
    """
    Extracts all names (including unconventional or stylized ones) from the given text.
    The function uses Ollama to return a comma-separated list of names and converts it into a Python list.
    
    Args:
        text (str): The input text from which names should be extracted.
    
    Returns:
        list: A list of names found in the text.
    
    Example:
        text = "The meeting was attended by John Doe, Jane Smith, and Mr. X."
        names = extract_names(text)
        print(names)
    """
    prompt = (
        "Extract all names of people from the following text. "
        "Ignore vehicle names, item names, and organizational names. "
        "Exclude words such as 'Caller', 'Complainant', and 'Key Holder' (or ' Kh ') but keep the name associated with 'Kh' if a key holder is mentioned. "
        "Return the names as a comma-separated list with each name trimmed of extra spaces.\n\n"
        f"Text:\n{text}"
    )

    
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
    names_output = response['message']['content']
    
    # Assuming the output is a comma-separated list, split and clean the names.
    names = [name.strip() for name in names_output.split(",") if name.strip()]
    
    return names

def split_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def convert_to_local_gps(utc_dt, timezone='America/New_York'):
    """Convert UTC datetime string to local time."""
    utc_dt = utc_dt[:-5]  # Remove timezone offset from string
    utc_dt = utc_dt.replace("T", " ")
    utc_dt = pd.to_datetime(utc_dt).tz_localize('UTC').astimezone(pytz.timezone(timezone))
    return utc_dt.strftime('%Y-%m-%d %H:%M:%S')

def convert_to_utc(local_time):
    """Convert local EST time to UTC time for API request."""
    est = pytz.timezone("America/New_York")
    utc = pytz.utc
    local_dt = est.localize(datetime.strptime(local_time, '%Y-%m-%dT%H:%M'))
    return local_dt.astimezone(utc).strftime('%Y-%m-%dT%H:%M')

def fix_time_string_regex(time_str):
    """Uses regex to replace the first '-' after 'T' with ':'."""
    return re.sub(r'(?<=T\d\d)-', ":", time_str, count=1)

def assign_shift(local_dt):
    """
    Given a pandas Timestamp (local datetime), assign a shift:
    - 1st Shift (Midnight): 22:00 to 06:00
    - 2nd Shift (Day): 06:00 to 14:00
    - 3rd Shift (Afternoon): 14:00 to 22:00
    """
    hour = local_dt.hour
    # Midnight shift covers hours 22-23 and 0-5
    if hour >= 22 or hour < 6:
        return "1st Shift"
    elif 6 <= hour < 14:
        return "2nd Shift"
    else:
        return "3rd Shift"
    
def round_to_nearest_hour(dt):
    # Round down if minutes < 30; otherwise, round up.
    if dt.minute < 30:
        return dt.replace(minute=0, second=0, microsecond=0)
    else:
        return dt.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    
def get_hourly_weather(start_date_local=None, end_date_local=None, excel_file="resources/weather_data.xlsx"):
    """
    Fetches hourly weather data from Meteostat for Canton, OH.
    - Loads existing data from an Excel file if available.
    - If no start/end date is provided, fetches from the last available timestamp in the file.
    - Converts timestamps for Excel compatibility (removes timezone).
    - Appends new data and removes duplicates.
    - Saves back to Excel.

    Args:
        start_date_local (str, optional): Start date in 'YYYY-MM-DD' format (LOCAL TIME). If None, auto-detects.
        end_date_local (str, optional): End date in 'YYYY-MM-DD' format (LOCAL TIME). If None, uses today's date.
        excel_file (str): Filepath to store the data.

    Returns:
        pd.DataFrame: Updated DataFrame with the new and existing weather data.

    Example:
        Example Usage: Get hourly weather data for date and for excel file to pull last date. It does not need
        to be assigned as a variable.
        weather_df = get_hourly_weather("2021-06-01", "2021-06-30")
        weather_df = get_hourly_weather()
    """

    # Define Canton, OH location
    canton = Point(40.799, -81.3784)

    # Define time zones
    eastern = pytz.timezone("US/Eastern")
    utc = pytz.utc

    # Load existing data if the Excel file exists
    existing_df = pd.DataFrame()  # Default to empty DataFrame if no file exists
    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file, parse_dates=["time_local"])
        existing_df["time_local"] = pd.to_datetime(existing_df["time_local"])

    # Auto-detect start date if not provided (use the last recorded time_local)
    if start_date_local is None and not existing_df.empty:
        last_recorded_time = existing_df["time_local"].max()
        start_date_local = last_recorded_time.strftime("%Y-%m-%d")
    elif start_date_local is None:
        start_date_local = (datetime.now(eastern) - timedelta(days=30)).strftime("%Y-%m-%d")  # Default to last 30 days

    # If no end date provided, use the current date
    if end_date_local is None:
        end_date_local = datetime.now(eastern).strftime("%Y-%m-%d")

    # Convert input local dates to datetime objects (with ET timezone)
    start_dt_local = eastern.localize(datetime.strptime(start_date_local, "%Y-%m-%d"), is_dst=None)
    end_dt_local = eastern.localize(datetime.strptime(end_date_local, "%Y-%m-%d"), is_dst=None) + timedelta(days=1)

    # Convert local time to UTC for Meteostat request
    start_dt_utc_naive = start_dt_local.astimezone(utc).replace(tzinfo=None)
    end_dt_utc_naive = end_dt_local.astimezone(utc).replace(tzinfo=None)

    # Fetch hourly weather data from Meteostat in UTC
    data = Hourly(canton, start=start_dt_utc_naive, end=end_dt_utc_naive)
    df = data.fetch()

    # Reset index to access 'time' as a column
    df.reset_index(inplace=True)

    # Convert 'time' column explicitly to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Handle AmbiguousTimeError during DST changes
    try:
        df["time_local"] = df["time"].dt.tz_localize(utc).dt.tz_convert(eastern)
    except pytz.AmbiguousTimeError:
        df["time_local"] = df["time"].dt.tz_localize(utc).dt.tz_convert(eastern, ambiguous="NaT")

    # Remove rows with NaT (caused by ambiguous times that cannot be resolved)
    df = df.dropna(subset=["time_local"])

    # Create 'weather_relationship' field (rounded local time for merging)
    # df["weather_relationship"] = df["time_local"].dt.round("h")
    # Handle rounding error due to ambiguous times (DST transitions)
    try:
        df["weather_relationship"] = df["time_local"].dt.round("h")
    except pytz.AmbiguousTimeError:
        # If AmbiguousTimeError occurs, try setting ambiguous times to NaT
        df["time_local"] = df["time_local"].dt.tz_localize(None)  # Remove timezone first
        df["weather_relationship"] = df["time_local"].dt.round("h")

    # Filter out any future timestamps
    current_utc_time = datetime.utcnow().replace(tzinfo=None)
    df = df[df["time"] <= current_utc_time]

    # Convert temperature & dew point from Celsius to Fahrenheit
    df["temp"] = df["temp"] * 9/5 + 32
    df["dwpt"] = df["dwpt"] * 9/5 + 32

    # Convert wind speed from km/h to mph
    df["wspd"] = df["wspd"] * 0.621371

    # Convert pressure from hPa to inHg
    df["pres"] = df["pres"] * 0.02953

    # Convert precipitation from mm to inches
    df["prcp"] = df["prcp"] * 0.0393701

    # Convert snow depth from cm to inches
    df["snow"] = df["snow"] * 0.393701

    # Remove timezone information for Excel compatibility
    df["time_local"] = df["time_local"].dt.strftime("%Y-%m-%d %H:%M:%S")  # Removes timezone offset

    # Rename columns for clarity
    df.rename(columns={
        "temp": "temp_f",
        "dwpt": "dwpt_f",
        "wspd": "wspd_mph",
        "pres": "pres_inHg",
        "prcp": "prcp_in",
        "snow": "snow_in"
    }, inplace=True)

    # Select relevant columns for merging
    df = df[["time_local", "temp_f", "dwpt_f", "rhum", "prcp_in", "wspd_mph", "wdir", "pres_inHg", "snow_in"]]

    # **Merge with existing data and remove duplicates**
    if not existing_df.empty:
        # Ensure both DataFrames have 'time_local' as a datetime object
        df["time_local"] = pd.to_datetime(df["time_local"])
        existing_df["time_local"] = pd.to_datetime(existing_df["time_local"])

        # Merge, remove duplicates, and sort
        combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=["time_local"]).sort_values("time_local")
    else:
        combined_df = df

    # Ensure 'time_local' is timezone-unaware for Excel compatibility
    combined_df["time_local"] = pd.to_datetime(combined_df["time_local"]).dt.tz_localize(None)
    combined_df["weather_relationship"] = pd.to_datetime(combined_df["time_local"]).dt.tz_localize(None)

    # **Save the updated DataFrame to Excel**
    combined_df.to_excel(excel_file, index=False)

    return combined_df

def filter_by_radius(df, lat_col, lon_col, ref_lat, ref_lon, radius_feet):
    """
    Filters a DataFrame to return rows where coordinates fall within a specified radius (in feet).
    
    Args:
        df (pd.DataFrame): The DataFrame containing latitude and longitude columns.
        lat_col (str): Column name for latitude in the DataFrame.
        lon_col (str): Column name for longitude in the DataFrame.
        ref_lat (float): Reference latitude (the center point).
        ref_lon (float): Reference longitude (the center point).
        radius_feet (float): Radius in feet to filter locations.

    Returns:
        pd.DataFrame: Filtered DataFrame with rows within the given radius.
    
    Example:
        # Filter the DataFrame to include only locations within 1000 feet of a reference point
        filtered_df = filter_by_radius(df, "Latitude", "Longitude", 40.7128, -74.0060, 1000)
        # Example usage:
        # Assume we want to load data between 2025-02-22T20:00 and 2025-02-23T02:00
        df_in_range = fn.load_gps_data_in_range("2025-02-23T02:01", "2025-02-23T03:10")
        test = fn.filter_by_radius(df_in_range, "latitude", "longitude", 40.785199, -81.414820, 2000)
        test.head(3)
    """

    # Convert columns to numeric and handle errors
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    # Remove NaN values
    df = df.dropna(subset=[lat_col, lon_col])

    # Ensure latitudes and longitudes are within valid ranges
    df = df[(df[lat_col].between(-90, 90)) & (df[lon_col].between(-180, 180))]

    # Convert feet to miles (1 mile = 5280 feet)
    radius_miles = radius_feet / 5280

    # Function to compute distance
    def is_within_radius(row):
        point = (row[lat_col], row[lon_col])
        ref_point = (ref_lat, ref_lon)
        return geodesic(ref_point, point).miles <= radius_miles

    # Apply the function and filter the DataFrame
    filtered_df = df[df.apply(is_within_radius, axis=1)]

    return filtered_df


## ----------------- GPS Functions ----------------- ##
def load_gps_data_in_range(request_start, request_end, folder="gps_data"):
    """
    Load GPS data from Excel files in a folder, but only load files whose time ranges (as indicated in the filename)
    overlap with the requested datetime range. The Excel files are assumed to have names in the format:
    
        gps_YYYY-MM-DDTHH-MM_YYYY-MM-DDTHH-MM.xlsx
    
    The function then filters the combined data to only include rows where the "datetime_local" column falls within
    the requested range.
    
    Args:
        request_start (str or datetime): Requested start time in 'YYYY-MM-DDTHH:MM' format (local time) or as a datetime.
        request_end (str or datetime): Requested end time in 'YYYY-MM-DDTHH:MM' format (local time) or as a datetime.
        folder (str): Path to the folder containing the Excel files.
        
    Returns:
        pd.DataFrame: A combined DataFrame of all matching data, filtered by "datetime_local".
    """
    # Convert request_start and request_end to datetime objects if they are strings.
    if isinstance(request_start, str):
        request_start_dt = datetime.strptime(request_start, '%Y-%m-%dT%H:%M')
    else:
        request_start_dt = request_start
    if isinstance(request_end, str):
        request_end_dt = datetime.strptime(request_end, '%Y-%m-%dT%H:%M')
    else:
        request_end_dt = request_end
    
    combined_df = pd.DataFrame()
    
    # List all files in the folder that follow the gps_*.xlsx naming pattern.
    for filename in os.listdir(folder):
        if filename.startswith("gps_") and filename.endswith(".xlsx"):
            try:
                # Expected format: gps_YYYY-MM-DDTHH-MM_YYYY-MM-DDTHH-MM.xlsx
                parts = filename.split("_")
                if len(parts) < 3:
                    continue  # Skip if filename doesn't conform
                
                # parts[1] is the start timestamp, parts[2] is the end timestamp (with .xlsx attached)
                file_start_raw = parts[1]  # e.g. "2025-02-22T20-00"
                file_end_raw = parts[2].replace(".xlsx", "")  # e.g. "2025-02-23T02-00"
                
                # Convert the time part from "HH-MM" to "HH:MM" while keeping the date intact.
                def fix_timestamp(ts):
                    # Split at 'T'
                    date_part, time_part = ts.split("T")
                    # Replace first hyphen in time part with colon
                    fixed_time = time_part.replace("-", ":", 1)
                    return f"{date_part}T{fixed_time}"
                
                file_start_str = fix_timestamp(file_start_raw)  # e.g. "2025-02-22T20:00"
                file_end_str = fix_timestamp(file_end_raw)        # e.g. "2025-02-23T02:00"
                
                file_start_dt = datetime.strptime(file_start_str, '%Y-%m-%dT%H:%M')
                file_end_dt = datetime.strptime(file_end_str, '%Y-%m-%dT%H:%M')
                
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
                continue

            # Check if the file's range overlaps with the requested range.
            # Overlap condition: file_end_dt >= request_start_dt and file_start_dt <= request_end_dt
            if file_end_dt >= request_start_dt and file_start_dt <= request_end_dt:
                try:
                    df = pd.read_excel(os.path.join(folder, filename), parse_dates=["datetime_local"])
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
                    continue

    # If data was loaded, filter by the requested range using the datetime_local column.
    if not combined_df.empty:
        combined_df["datetime_local"] = pd.to_datetime(combined_df["datetime_local"])
        combined_df = combined_df[
            (combined_df["datetime_local"] >= request_start_dt) &
            (combined_df["datetime_local"] <= request_end_dt)
        ]
    return combined_df

# def initialize_globals(credentials):
def initialize_globals():
    """Initialize global GPS configuration settings."""
    # Sentinel DNA API Details
    GPSConfig.BASE_URL = 'https://datatransfer.sentineldna.com'
    GPSConfig.AUTHORIZE_ENDPOINT = '/v1/security/signin'
    GPSConfig.PORT = "443"
    # GPSConfig.SOURCE_ID = credentials['GPS_SOURCE_ID']
    GPSConfig.SOURCE_ID = os.environ['GPS_SOURCE_ID']

    # GPSConfig.GPSCREDENTIALS = {
    #     'signInName': credentials['GPS_USERNAME'],
    #     'password': credentials['GPS_PASSWORD']
    # }
    GPSConfig.GPSCREDENTIALS = {
        'signInName': os.environ['GPS_USERNAME'],
        'password': os.environ['GPS_PASSWORD']
    }

    GPSConfig.HEADERS_TEMPLATE = {
        'Content-type': 'application/json; charset=utf-8',
        'Accept': 'application/json'
    }

    # Reset authentication token details
    GPSConfig.TOKEN = None
    GPSConfig.TOKEN_TIMESTAMP = None
    GPSConfig.TOKEN_EXPIRY = 59 * 60  # 59 minutes

    # Ensure the GPS data directory exists
    os.makedirs(GPSConfig.GPS_DATA_DIR, exist_ok=True)

def get_token():
    """Request a new authentication token."""
    authorize_url = f"{GPSConfig.BASE_URL}{GPSConfig.AUTHORIZE_ENDPOINT}"
    
    response = requests.post(
        authorize_url,
        headers=GPSConfig.HEADERS_TEMPLATE,
        json=GPSConfig.GPSCREDENTIALS,
        verify=False
    )
    
    response.raise_for_status()
    
    # Update token and timestamp directly in the class
    GPSConfig.TOKEN = response.json().get('jwt')
    GPSConfig.TOKEN_TIMESTAMP = time.time()

def is_token_expired():
    """Check if the authentication token has expired."""
    return (
        GPSConfig.TOKEN is None or 
        (time.time() - GPSConfig.TOKEN_TIMESTAMP) > GPSConfig.TOKEN_EXPIRY
    )

def get_headers():
    """Retrieve the authorization headers, refreshing the token if needed."""
    if is_token_expired():
        get_token()  # This will update GPSConfig.TOKEN and GPSConfig.TOKEN_TIMESTAMP

    return {
        'Authorization': f'Bearer {GPSConfig.TOKEN}',
        'Content-type': 'application/json; charset=utf-8',
        'Accept': 'application/json'
    }

def get_last_datetime():
    """Retrieve the last collected datetime from the most recent GPS data file."""
    try:
        # Get all relevant files
        files = [f for f in os.listdir(GPSConfig.GPS_DATA_DIR) if f.startswith("gps_") and f.endswith(".xlsx")]
        
        # print(f"DEBUG: Found files: {files}")  # Debugging: List of files found

        if files:
            base_files = {}

            for f in files:
                # Remove file extension
                filename_no_ext = f.replace(".xlsx", "")

                # If the filename has _partX, remove it
                if "_part" in filename_no_ext:
                    base_name = "_".join(filename_no_ext.split("_")[:-1])  # Remove "_partX"
                    # print(f"DEBUG: Stripped chunk suffix from {f} -> {base_name}")
                else:
                    base_name = filename_no_ext  # Keep as is if no "_partX"
                    # print(f"DEBUG: No chunk suffix found in {f}")

                # Keep track of the latest file for each base name
                if base_name not in base_files or f > base_files[base_name]:
                    base_files[base_name] = f
                    # print(f"DEBUG: Latest file for {base_name} -> {f}")

            # Get the latest overall file
            latest_file = sorted(base_files.values(), reverse=True)[0]
            # print(f"DEBUG: Selected latest file: {latest_file}")

            # Extract datetime from the full filename, ignoring _partX
            match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2})_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2})", latest_file)
            if match:
                last_dt_str = match.group(2)  # Get the ending datetime
                # print(f"DEBUG: Extracted last datetime string: {last_dt_str}")

                # Ensure proper format
                last_dt_str = fix_time_string_regex(last_dt_str)  
                # print(f"DEBUG: Fixed datetime string: {last_dt_str}")

                return last_dt_str  # Returns in UTC format: "YYYY-MM-DDTHH:MM"

        # If no files exist, pull data from the last 24 hours
        fallback_time = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M')
        # print(f"DEBUG: No files found, using fallback time: {fallback_time}")
        return fallback_time

    except Exception as e:
        # print(f"ERROR: Retrieving last datetime failed: {e}")
        fallback_time = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M')
        # print(f"DEBUG: Using fallback time due to error: {fallback_time}")
        return fallback_time

def get_data(from_time, to_time):
    """Retrieve GPS data from Sentinel DNA API."""
    data_request_url = f"{GPSConfig.BASE_URL}:{GPSConfig.PORT}/v1/location?sourceid={GPSConfig.SOURCE_ID}&from={from_time}&to={to_time}"
    response = requests.get(data_request_url, headers=get_headers(), verify=False)
    response.raise_for_status()
    return pd.DataFrame(response.json())

# def collect_gps_data(credentials, start_dt=None, end_dt=None):
def collect_gps_data(start_dt=None, end_dt=None):
    """
    Collect GPS data in 5-minute intervals and save in chunks if the dataset is too large.
    - Uses `start_dt` and `end_dt` if provided (in local EST format).
    - If `start_dt` is None, retrieves the last known datetime.
    - Converts times to UTC for API requests.
    - Saves the collected data in Excel files in chunks if necessary.
    """
    # Excel sheet row limit
    MAX_ROWS = 1_040_000  # Slightly below 1,048,576 to leave space for headers

    # Suppress InsecureRequestWarning
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Initialize global variables
    # initialize_globals(credentials)
    initialize_globals()

    # Determine the start time
    if start_dt is None:
        start_dt = get_last_datetime()  # Get last recorded timestamp
    if end_dt is None:
        end_dt = datetime.now().strftime('%Y-%m-%dT%H:%M')  # Use current time

    # Convert provided local times to UTC for API
    start_time = convert_to_utc(start_dt)
    end_time = convert_to_utc(end_dt)

    # Initialize an empty list to store dataframes
    dataframes = []

    # Loop through 5-minute intervals
    current_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M')
    end_time_dt = datetime.strptime(end_time, '%Y-%m-%dT%H:%M')

    while current_time < end_time_dt:
        next_time = current_time + timedelta(minutes=5)

        # Convert back to string format
        start_str = current_time.strftime('%Y-%m-%dT%H:%M')
        end_str = next_time.strftime('%Y-%m-%dT%H:%M')

        try:
            df = get_data(start_str, end_str)
            if not df.empty:
                df['datetime_local'] = df['receivedtime'].apply(convert_to_local_gps)
                dataframes.append(df)
        except Exception as e:
            print(f"Error fetching data for {start_str} to {end_str}: {e}")

        current_time = next_time  # Move to next interval

    # Combine all dataframes if any data was retrieved
    if dataframes:
        result_df = pd.concat(dataframes, ignore_index=True)

        # Create base filename
        base_filename = f"{GPSConfig.GPS_DATA_DIR}/gps_{start_dt.replace(':', '-')}_{end_dt.replace(':', '-')}"
        
        # Split and save in chunks
        num_chunks = (len(result_df) // MAX_ROWS) + 1  # Determine number of chunks
        for i in range(num_chunks):
            chunk_df = result_df.iloc[i * MAX_ROWS: (i + 1) * MAX_ROWS]  # Slice data
            
            if not chunk_df.empty:
                filename = f"{base_filename}_part{i+1}.xlsx" if num_chunks > 1 else f"{base_filename}.xlsx"
                chunk_df.to_excel(filename, index=False, header=True)
                print(f"Chunk {i+1} saved to {filename}")

        # Delete files older than 25 days
        delete_old_files(GPSConfig.GPS_DATA_DIR, 25)
        return result_df  # Return full dataframe

    print("No new data retrieved.")
    return None

def search_gps_area(dt_start=None, dt_end=None, lat=None, lon=None, radius=1000, folder="resources/gps_data", export_folder="resources/gps_search_exports"):
    # Prompt individually for any missing values
    if dt_start is None:
        dt_start = input("Enter start datetime (YYYY-MM-DDTHH:MM): ")
    if dt_end is None:
        dt_end = input("Enter end datetime (YYYY-MM-DDTHH:MM): ")
    if lat is None:
        lat = float(input("Enter latitude: ex(40.123456)"))
    if lon is None:
        lon = float(input("Enter longitude: ex(-81.123456)"))

    # Load GPS data within the given date range
    gps_df = load_gps_data_in_range(dt_start, dt_end, folder=folder)

    # Filter data based on the given latitude, longitude, and radius
    in_area = filter_by_radius(gps_df, "latitude", "longitude", lat, lon, radius)

    # Determine file name based on whether the DataFrame is empty
    file_prefix = "EMPTY_" if in_area.empty else ""
    file_name = f"{file_prefix}GPSDATA_{dt_start}_{dt_end}.xlsx"
    file_path = f"{export_folder}/{file_name}".replace(":", "-")

    if in_area.empty:
        print(f"No data found within the specified radius of {radius} feet from ({lat}, {lon}) between {dt_start} and {dt_end}.")
    else:
        print(f"Data found within the specified radius of {radius} feet from ({lat}, {lon}) between {dt_start} and {dt_end}.")
        print(f"Exporting data to {file_path}...")
        
    # Export to Excel
    in_area.to_excel(file_path, index=False)

 
## ----------------- Calls For Service Functions ----------------- ##
def fetch_calls_for_service(credentials, ori=None, start_date=None, end_date=None, limit=1000, data_type="All"):
    conn = server_connect(credentials['CAD_SERVER'], credentials['CAD_UID'], credentials['CAD_PWD'], credentials['CAD_DB'])

    # Apply limit only if no date filters are provided
    limit_clause = f"TOP {limit}" if start_date is None and end_date is None else ""

    # Base SELECT fields (common for both "All" and "Stats")
    base_fields = f"""
        {limit_clause}
        Call.Call.CallID,
        Call.Incident.IncidentNumber, 
        Call.Incident.ORIID, 
        ReadOnly.ORI.ORI, 
        Call.Call.CreateDatetime, 
        DateAdd(hour, -5, Call.Call.CreateDatetime) as LocalDatetime,
        ReadOnly.ORI.Department, 
        General.ValidationSetEntry.EntryValue as CallType,
        Call.Call.CommonName,
        Geography.Address.AddressID, 
        Geography.Address.HouseNumber, 
        Geography.Address.PrefixDirectional, 
        Geography.Address.PrefixType, 
        Geography.Address.StreetName, 
        Geography.Address.StreetType, 
        Geography.Address.StreetDirectional,
        Geography.Address.XStreetName, 
        Geography.Address.XStreetType, 
        Geography.Address.XStreetDirectional,
        Geography.Address.City,
        Geography.Address.Qualifier as APT,
        Geography.Address.LatitudeY,
        Geography.Address.LongitudeX
    """

    # Additional fields for "All" data
    additional_fields = """
        , cn.notes,
        ds.dispo,
        cp.involved_people,
        cv.vehicles
    """

    # Base JOINs (common for both "All" and "Stats")
    base_joins = """
        FROM 
            Call.Call 
            INNER JOIN Call.Incident ON Call.Call.CallID = Call.Incident.CallID
            INNER JOIN ReadOnly.ORI ON Call.Incident.ORIID = ReadOnly.ORI.ORIID
            INNER JOIN General.ValidationSetEntry ON General.ValidationSetEntry.EntryID = Call.Incident.vsIncidentType
            INNER JOIN General.CallType ON Call.Call.CallTypeID = General.CallType.CallTypeID
            INNER JOIN Geography.Address ON Call.Call.CurrentAddressID = Geography.Address.AddressID
    """

    # Additional JOINs for "All" data
    additional_joins = """
        LEFT JOIN (
            SELECT
                CallID,
                STRING_AGG(Narrative, ' ') AS notes
            FROM
                Call.CallNarrative
            GROUP BY
                CallID
        ) AS cn ON Call.Call.CallID = cn.CallID
        LEFT JOIN (
            SELECT
                Call.CallDisposition.CallID,
                STRING_AGG(General.ValidationSetEntry.EntryDescription, ', ') AS dispo
            FROM
                Call.Call
                LEFT JOIN Call.CallDisposition ON Call.Call.CallID = Call.CallDisposition.CallID
                LEFT JOIN General.ValidationSetEntry ON 
                Call.CallDisposition.vsDisposition = General.ValidationSetEntry.EntryID
            GROUP BY
                Call.CallDisposition.CallID
        ) AS ds ON Call.Call.CallID = ds.CallID
        LEFT JOIN (
            SELECT
                CallID,
                STRING_AGG(
                    CONCAT(
                        CallPerson.LastName, ', ', 
                        CallPerson.FirstName, ' (', 
                        FORMAT(CallPerson.DateOfBirth, 'yyyy-MM-dd'),')'
                    ),
                    '\n'
                ) AS involved_people
            FROM
                Call.CallPerson
            GROUP BY
                CallID
        ) AS cp ON Call.Call.CallID = cp.CallID
        LEFT JOIN (
            SELECT
                CallID,
                STRING_AGG(LicenseNumber, ', ') AS vehicles
            FROM
                Call.CallVehicle
            GROUP BY
                CallID
        ) AS cv ON Call.Call.CallID = cv.CallID
    """

    # Construct query based on data_type
    if data_type == "All":
        query = f"SELECT {base_fields} {additional_fields} {base_joins} {additional_joins}"
    else:  # "Stats" mode, excluding the additional JOINs and fields
        query = f"SELECT {base_fields} {base_joins}"

    # Conditionally add WHERE clause if filters are provided
    where_clauses = []
    params = []

    if ori:
        where_clauses.append("ReadOnly.ORI.ORI = ?")
        params.append(ori)

    if start_date and end_date:
        # where_clauses.append("Call.Call.CreateDatetime BETWEEN ? AND ?")
        where_clauses.append("DateAdd(hour, -5, Call.Call.CreateDatetime) BETWEEN ? AND ?")
        params.extend([start_date, end_date])

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    # Only apply ORDER BY when limiting results
    if start_date is None and end_date is None:
        query += " ORDER BY Call.Call.CreateDatetime DESC"

    return pd.read_sql_query(query, conn, params=params)

def count_all_dispo_items(df: pd.DataFrame, separator: str = ',') -> Counter:
    """
    Splits the 'dispo' column (assumed to be a string of items separated by the specified separator)
    and returns a Counter object with the frequency of each item.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'dispo' column.
        separator (str): The character that separates the items in the 'dispo' string. Default is ','.
    
    Returns:
        Counter: A Counter object with each item and its count.

    Example:
        counts = count_all_dispo_items(cfs_df)
        print(counts)
    """
    # Drop missing values and split each string into a list of trimmed items.
    dispo_lists = df['dispo'].dropna().apply(lambda x: [item.strip() for item in x.split(separator)])
    # Flatten the lists into a single list of items.
    all_items = [item for sublist in dispo_lists for item in sublist if item]
    # Return the count for each item.
    return Counter(all_items)

def extract_report_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataframe for records where ORI is 'OH0760400' and dispo is 'Report Taken'.
    Uses the two-digit year from "LocalDatetime" to help identify a report number in the 'notes'
    field that follows the pattern (two digits, optional hyphen, five digits).
    
    If the 'CallType' contains "accident" (case insensitive), the extracted number is labeled as
    an AccidentNumber; otherwise, it is labeled as an IncidentNumber.
    
    If no report or accident number is found in the notes, the function returns None for that record.
    
    Returns a new dataframe with columns: "CallID", "LocalDatetime", "FullAddress", "CallType",
    "IncidentNumber", "ReportNumber", and "NumberType".
    """
    # Filter for your department's records and only rows where dispo is "Report Taken"
    df_filtered = df[(df['ORI'] == 'OH0760400') & (df['dispo'].str.contains('Report Taken', na=False))].copy()
    
    def extract_report_number(row):
        notes = row['notes']
        # Get two-digit year from LocalDatetime
        local_year = pd.to_datetime(row['LocalDatetime']).strftime('%y')
        if isinstance(notes, str):
            # Find all matches of the pattern: two digits, optional hyphen, five digits
            matches = re.findall(r'\b(\d{2})-?(\d{5})\b', notes)
            if matches:
                # First, try to pick a match that begins with the local year.
                for match in matches:
                    if match[0] == local_year:
                        return match[0] + match[1]
                # If no match starts with the local year, return the first match.
                return matches[0][0] + matches[0][1]
        # If no match is found, return None
        return None
    
    # Remove hyphen and first two numbers from IncidentNumber prior to returning
    df_filtered['IncidentNumber'] = df_filtered['IncidentNumber'].str.replace('-', '').str[2:]

    # Apply the extraction function row-wise
    df_filtered['ReportNumber'] = df_filtered.apply(extract_report_number, axis=1)
    
    # Determine the number type based on the CallType field
    df_filtered['NumberType'] = df_filtered['CallType'].apply(
        lambda ct: "AccidentNumber" if isinstance(ct, str) and "accident" in ct.lower() else "IncidentNumber"
    )
    
    # Select and return only the desired columns
    result = df_filtered[['CallID', 'LocalDatetime', 'FullAddress', 'CallType', 'IncidentNumber', 'ReportNumber', 'NumberType']]
    result.to_excel("CFS_Report_list.xlsx", index=False, header=True)
    return result

def preprocess_calls(df, timezone='America/New_York'):
    # Rename 'Name' to 'CallType'
    df.rename(columns={"Name": "CallType"}, inplace=True)

    # Convert datetime fields to uniform format
    # df["LocalDatetime"] = pd.to_datetime(df["LocalDatetime"], errors="coerce")
    df["CreateDatetime"] = pd.to_datetime(df["CreateDatetime"], errors="coerce")

    # Convert LocalDatetime from UTC to local dynamically
    local_tz = pytz.timezone(timezone)
    # df["LocalDatetime"] = df["CreateDatetime"].dt.tz_localize('UTC').dt.tz_convert(local_tz)
    df["LocalDatetime"] = df["CreateDatetime"].dt.tz_localize('UTC').dt.tz_convert(local_tz).dt.tz_localize(None)

    # Round LocalDatetime to the nearest hour
    df['weather_relationship'] = df['LocalDatetime'].apply(round_to_nearest_hour)

    # Convert HouseNumber to an integer string if it's effectively an integer,
    # otherwise just convert it to a string.
    df["HouseNumber"] = df["HouseNumber"].apply(
        lambda x: str(int(x)) if pd.notna(x) and isinstance(x, float) and x.is_integer() else str(x)
    )

    # Assign Shift based on LocalDatetime
    df["Shift"] = df["LocalDatetime"].apply(assign_shift)
    
    # Generate FullAddress
    df["FullAddress"] = np.where(
        # Check if HouseNumber equals -1 OR if both StreetName and XStreetName are blank
        (df["HouseNumber"].astype(str).str.strip() == "-1") |
        ((df["StreetName"].fillna('').str.strip() == "") & (df["XStreetName"].fillna('').str.strip() == "")),
        "Not Listed",
        np.where(
            # Check if HouseNumber is valid (not NaN, not empty, not "nan")
            df["HouseNumber"].notna() & 
            (df["HouseNumber"].astype(str).str.strip() != "") & 
            (df["HouseNumber"].astype(str).str.strip() != "nan"),
            # If valid, build address using HouseNumber and primary street info
            df["HouseNumber"].astype(str) + " " +
            df["StreetName"].fillna('') + " " +
            df["StreetType"].fillna('') + " " +
            df["StreetDirectional"].fillna(''),
            # Otherwise, combine primary street info with cross street info using "&"
            df["StreetName"].fillna('') + " " +
            df["StreetType"].fillna('') + " " +
            df["StreetDirectional"].fillna('') + " & " +
            df["XStreetName"].fillna('') + " " +
            df["XStreetType"].fillna('') + " " +
            df["XStreetDirectional"].fillna('')
        )
    )


    # Replace NaN values in text fields with "Unknown"
    text_columns = ["CallType", "notes", "dispo", "involved_people", "vehicles"]
    df[text_columns] = df[text_columns].fillna("Unknown")

    # Truncate notes field to 500 characters
    # df["notes"] = df["notes"].astype(str).str[:500]
    df["notes_chunks"] = df["notes"].apply(lambda x: split_text(x) if len(x) > 500 else [x])

    return df

def update_daily_summary(cfs_data, csv_filename="call_type_daily_summary.csv", timezone='America/New_York'):
    """
    Updates or initializes the call type daily summary file.

    Parameters:
    - cfs_data: DataFrame containing Calls for Service data with only ORI, CreateDatetime, and CallType.
    - csv_filename: The CSV file to store the daily summary.
    - timezone: The local timezone for datetime conversion.
    """

    # Ensure required columns exist
    required_columns = {"ORI", "CreateDatetime", "CallType"}
    if not required_columns.issubset(set(cfs_data.columns)):
        raise ValueError("Missing required columns in data")

    # Convert CreateDatetime from UTC to local time
    cfs_data["CreateDatetime"] = pd.to_datetime(cfs_data["CreateDatetime"], errors="coerce")
    local_tz = pytz.timezone(timezone)
    cfs_data["LocalDatetime"] = cfs_data["CreateDatetime"].dt.tz_localize('UTC').dt.tz_convert(local_tz)

    # Assign shift based on local time
    cfs_data["Shift"] = cfs_data["LocalDatetime"].apply(assign_shift)

    # Extract just the date (removing time part)
    cfs_data["Date"] = cfs_data["LocalDatetime"].dt.date

    # Select final required columns for summary
    cfs_summary = cfs_data.groupby(["ORI", "Date", "Shift", "CallType"]).size().reset_index(name="Total_Calls")

    # Check if the CSV already exists
    if os.path.exists(csv_filename):
        existing_summary = pd.read_csv(csv_filename, parse_dates=["Date"])
        
        # # Get the last recorded date
        # last_date = existing_summary["Date"].max()

        # # Only keep new data (from the last recorded date up to yesterday)
        # new_data = cfs_summary[(cfs_summary["Date"] > last_date) & (cfs_summary["Date"] < pd.Timestamp.today().date())]

        last_date = existing_summary["Date"].max().date()  # Convert to date object

        # Ensure Date column is also a datetime.date for valid comparison
        cfs_summary["Date"] = pd.to_datetime(cfs_summary["Date"]).dt.date

        # Only keep new data (from the last recorded date up to yesterday)
        new_data = cfs_summary[(cfs_summary["Date"] > last_date) & (cfs_summary["Date"] < pd.Timestamp.today().date())]
    else:
        # No existing file, initialize with full available data
        new_data = cfs_summary

    # If no new data to process, exit
    if new_data.empty:
        print("No new data to process.")
        return

    # Append new data to the CSV (creating file if necessary)
    new_data.to_csv(csv_filename, mode="a", header=not os.path.exists(csv_filename), index=False)

    print(f"Updated {csv_filename} with {len(new_data)} new records.")

def load_summary_data(csv_filename="call_type_daily_summary.csv", years=5):
    """Loads the call type daily summary CSV and filters the last 'years' years of data."""
    
    # Read CSV (parse Date column as datetime)
    df = pd.read_csv(csv_filename, parse_dates=["Date"])

    # Convert Date column to date type to match comparison
    df["Date"] = df["Date"].dt.date  
    
    # Get the current year and today's date
    today = pd.Timestamp.today().date()
    current_year = today.year
    
    # Filter for the past 'years' worth of data
    df = df[df["Date"] >= pd.Timestamp(f"{current_year - years}-01-01").date()]
    
    return df

def compute_5yr_ytd_average(df):
    """Calculates the 5-year average year-to-date (YTD) call counts, using the last available date for comparison."""
    
    today = pd.Timestamp.today().date()
    current_year = today.year

    # Ensure Date is in datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Find last available date in the current year
    last_available_date = df[df["Date"].dt.year == current_year]["Date"].max()

    # Use the last available date instead of today's date
    ytd_end_date = last_available_date.strftime("%m-%d") if pd.notna(last_available_date) else today.strftime("%m-%d")

    # Convert Date to MM-DD format
    df["Month-Day"] = df["Date"].dt.strftime("%m-%d")  

    # Filter for only YTD dates (Jan 1 - last available date) across the last 5 years
    ytd_data = df[df["Month-Day"] <= ytd_end_date]

    # Exclude the current year's data from the 5-year average calculation
    ytd_data = ytd_data[ytd_data["Date"].dt.year < current_year]

    # Group by Year, ORI, and CallType to get totals per year
    ytd_totals = ytd_data.groupby([ytd_data["Date"].dt.year, "ORI", "CallType"])["Total_Calls"].sum().reset_index()

    # Compute 5-year average per ORI and CallType
    five_year_avg = ytd_totals.groupby(["ORI", "CallType"])["Total_Calls"].mean().reset_index()
    five_year_avg.rename(columns={"Total_Calls": "5yr_YTD_Avg"}, inplace=True)
    
    return five_year_avg

def compute_5yr_weekly_average(df):
    """Calculates the 5-year average for the last available week's call counts, excluding the current year."""
    
    # Get current year
    current_year = pd.Timestamp.today().year

    # Ensure Date is in datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Find the last available week from the dataset
    last_available_week = df[df["Date"].dt.year == current_year]["Date"].dt.isocalendar().week.max()

    if pd.isna(last_available_week):  # If no current year data exists
        print("No current year data available for weekly comparison.")
        return None

    # Get ISO week number for filtering
    df["Week"] = df["Date"].dt.isocalendar().week

    # Filter for past 5 years excluding the current year, using the last available week
    weekly_data = df[(df["Week"] == last_available_week) & (df["Date"].dt.year < current_year)]

    # Group by Year, ORI, and CallType to get totals per week
    weekly_totals = weekly_data.groupby([df["Date"].dt.year, "ORI", "CallType"])["Total_Calls"].sum().reset_index()

    # Compute 5-year average per ORI and CallType
    five_year_avg_week = weekly_totals.groupby(["ORI", "CallType"])["Total_Calls"].mean().reset_index()
    five_year_avg_week.rename(columns={"Total_Calls": "5yr_Weekly_Avg"}, inplace=True)
    
    return five_year_avg_week

def compare_current_to_5yr(df, ytd_avg, weekly_avg):
    """Merges current year data with 5-year averages for comparison, using the last available date."""
    
    current_year = pd.Timestamp.today().year
    
    # Find the last available date for the current year
    last_date = df[df["Date"].dt.year == current_year]["Date"].max()
    
    if pd.isna(last_date):  # If no current year data exists
        print("No current year data available for comparison.")
        return None, None
    
    # Compute current YTD total (from Jan 1 to last available date)
    ytd_current = df[(df["Date"].dt.year == current_year) & (df["Date"] <= last_date)]
    ytd_current = ytd_current.groupby(["ORI", "CallType"])["Total_Calls"].sum().reset_index()

    # Merge with YTD 5-year average
    ytd_comparison = ytd_current.merge(ytd_avg, on=["ORI", "CallType"], how="left")
    ytd_comparison["% Change YTD"] = ((ytd_comparison["Total_Calls"] - ytd_comparison["5yr_YTD_Avg"]) / 
                                       ytd_comparison["5yr_YTD_Avg"]) * 100
    
    # Find last available week number (ISO format)
    last_week = df[df["Date"].dt.year == current_year]["Date"].dt.isocalendar().week.max()
    
    # Compute current year's total for that week
    weekly_current = df[(df["Date"].dt.year == current_year) & (df["Date"].dt.isocalendar().week == last_week)]
    weekly_current = weekly_current.groupby(["ORI", "CallType"])["Total_Calls"].sum().reset_index()

    # Merge with Weekly 5-year average
    weekly_comparison = weekly_current.merge(weekly_avg, on=["ORI", "CallType"], how="left")
    weekly_comparison["% Change Weekly"] = ((weekly_comparison["Total_Calls"] - weekly_comparison["5yr_Weekly_Avg"]) / 
                                            weekly_comparison["5yr_Weekly_Avg"]) * 100

    return ytd_comparison, weekly_comparison

def compute_total_cfs_past_21_days(df):
    """Returns the total Calls for Service (CFS) for the past 21 days."""
    
    # Ensure Date is in datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Get the last available date
    last_available_date = df["Date"].max()
    
    if pd.isna(last_available_date):
        print("No available data.")
        return None

    # Get the start date (21 days before last available date)
    start_date = last_available_date - pd.Timedelta(days=21)

    # Filter for the past 21 days
    past_21_days_data = df[(df["Date"] >= start_date) & (df["Date"] <= last_available_date)]
    
    # Group by Date and sum total calls
    total_calls_per_day = past_21_days_data.groupby("Date")["Total_Calls"].sum().reset_index()

    return total_calls_per_day

def compute_total_cfs_past_10_weeks(df):
    """Returns the total Calls for Service (CFS) for the past 10 full weeks by extracting data from the past 70 days."""

    # Ensure Date is in datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Find the last available date
    last_available_date = df["Date"].max()
    
    if pd.isna(last_available_date):
        print("No available data.")
        return None

    # Determine the start date (70 days before last available date)
    start_date = last_available_date - pd.Timedelta(days=70)

    # Filter for the past 70 days
    past_70_days_data = df[(df["Date"] >= start_date) & (df["Date"] <= last_available_date)].copy()

    # Extract Year & Week Number dynamically
    past_70_days_data["Year"] = past_70_days_data["Date"].dt.year
    past_70_days_data["Week"] = past_70_days_data["Date"].dt.isocalendar().week

    # Group by Year-Week combination and sum total calls
    total_calls_per_week = past_70_days_data.groupby(["Year", "Week"])["Total_Calls"].sum().reset_index()

    # Sort by Year and Week to ensure correct chronological order
    total_calls_per_week = total_calls_per_week.sort_values(by=["Year", "Week"]).reset_index(drop=True)

    # Ensure we only return exactly 10 full weeks
    return total_calls_per_week.tail(10)

def compute_total_cfs_ytd(df):
    """Calculates total Calls for Service (CFS) YTD vs. 5-year average YTD, 
       includes the previous year's YTD total, and also tracks 
       the past 7 days (current week) vs. the prior 7 days (last week) with percentage change.
    """

    today = pd.Timestamp.today().date()
    current_year = today.year
    previous_year = current_year - 1

    # Ensure "Date" is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])  # Convert only if necessary

    # Find last available date in the current year's data
    last_available_date = df[df["Date"].dt.year == current_year]["Date"].max()
    if pd.isna(last_available_date):
        print("No current year data available.")
        return None

    # Get month-day format for filtering
    ytd_end_date = last_available_date.strftime("%m-%d")

    # Extract Month-Day for filtering
    df["Month-Day"] = df["Date"].dt.strftime("%m-%d")

    # Filter for current year's YTD data (Jan 1 - last available date)
    ytd_data_current_year = df[(df["Month-Day"] <= ytd_end_date) & (df["Date"].dt.year == current_year)]
    current_ytd_total = ytd_data_current_year["Total_Calls"].sum()

    # Filter for previous year's YTD data (same time frame)
    ytd_data_previous_year = df[(df["Month-Day"] <= ytd_end_date) & (df["Date"].dt.year == previous_year)]
    previous_ytd_total = ytd_data_previous_year["Total_Calls"].sum()

    # Compute 5-year average total CFS YTD (excluding current year)
    five_year_ytd_avg = (
        df[(df["Month-Day"] <= ytd_end_date) & (df["Date"].dt.year < current_year)]
        .groupby(df["Date"].dt.year)["Total_Calls"]
        .sum()
        .mean()
    )

    # Calculate percentage change YTD vs. last year
    if previous_ytd_total > 0:
        ytd_change_vs_last_year = ((current_ytd_total - previous_ytd_total) / previous_ytd_total) * 100
    else:
        ytd_change_vs_last_year = 0  # Avoid division by zero

    # Calculate percentage change YTD vs. 5-year average
    ytd_change_vs_5yr = ((current_ytd_total - five_year_ytd_avg) / five_year_ytd_avg) * 100

    ## ---- Calculate Current Week and Last Week ---- ##

    # Define date ranges
    start_current_week = last_available_date - pd.Timedelta(days=6)  # Last 7 days including last_available_date
    start_last_week = start_current_week - pd.Timedelta(days=7)  # Prior 7 days

    # Filter data
    current_week_data = df[(df["Date"] >= start_current_week) & (df["Date"] <= last_available_date)]
    last_week_data = df[(df["Date"] >= start_last_week) & (df["Date"] < start_current_week)]  # Exclude current week

    # Compute total calls
    current_week_total = current_week_data["Total_Calls"].sum()
    last_week_total = last_week_data["Total_Calls"].sum()

    # Calculate percentage change from Last Week to This Week
    if last_week_total > 0:
        week_change = ((current_week_total - last_week_total) / last_week_total) * 100
    else:
        week_change = 0  # Avoid division by zero

    ## ---- Return Results ---- ##
    ytd_details = {
        "Current_YTD": current_ytd_total, 
        "Previous_YTD": previous_ytd_total,
        "% Change YTD vs. Last Year": ytd_change_vs_last_year,
        "5yr_YTD_Avg": five_year_ytd_avg, 
        "% Change YTD vs. 5-Year Avg": ytd_change_vs_5yr,
        "Current Week": current_week_total,
        "Last Week": last_week_total,
        "% Change Week-to-Week": week_change
    }
    return ytd_details
