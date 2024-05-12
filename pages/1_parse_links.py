import pandas as pd
import os
import requests
import streamlit as st
from bs4 import BeautifulSoup as bs

URL = 'https://rozetker.kz/avtomobili'  # link to parse
page = 2
links_data = []  # List to store dictionaries of URLs

st.title("There we parse Data from Rozetker.kz")
st.subheader("Firstly parse links to our cars parameter page")

with st.echo():
    while page <= 2:  # parse 100 pages, you may adjust it
        response = requests.get(URL)  # send HTTP GET request
        if response.status_code == 200:  # check if request was successful
            content = response.text
            soup = bs(content, 'html.parser')

            link_contents = soup.findAll('div', attrs={'class': 'it-view-list'})  # get all div where class it_view_list

            # Loop through each div with class 'it-view-list'
            for link_content in link_contents:
                # Find all divs with class 'j-item it-list-item'
                items = link_content.find_all('div', class_='j-item it-list-item')
                # Loop through each found item
                for item in items:
                    # Find the anchor tag within the item
                    a_tag = item.find('a', class_='it-img-box')
                    # Extract the href attribute and append it to the list
                    if a_tag:
                        href = a_tag.get('href')
                        links_data.append({'URL': href})

            # Update the URL for the next page
            URL = 'https://rozetker.kz/avtomobili/?lt=list&page=' + str(page)
            page += 1
        else:
            print(f"Failed to retrieve page {page}. Status code: {response.status_code}")
            break

# Create DataFrame from the list of dictionaries
links = pd.DataFrame(links_data)

# Save the extracted links to a CSV file
links.to_csv("parse_links", index=False)

st.write("## Extracted URLs")
st.table(links.iloc[0:10])


st.subheader("Then parse from links to our cars parameter page")
st.code("""def car_data(url):
    response = requests.get(url)
    time.sleep(1)
    soup = BeautifulSoup(response.text, 'html.parser')
    properties = soup.find_all('div', class_='vw-dynprops-item')
    about_dict = {}
    cars = soup.findAll('div', attrs={'class': 'l-content'})
    
    car_name = car_body = car_year = car_transmission = car_mileage = car_volume = car_drive = car_steering_wheel = car_customs_clearance = car_color = 'Null'
    
    for car in cars:
        # Extract car_name and car_price
        car_name = car.find('h1', class_='l-page-title', itemprop='name').text.strip().split(',', 1)[0].split(',', 1)[0].split('(', 1)[0]
        body_price = car.find("div", attrs={'class': 'vw-price-box c-shadow-overflow'})
        price = body_price.find("span", class_="vw-price-num").text.strip()
        car_price = int(''.join(filter(str.isdigit, price)))

    for prop in properties:
        attribute = prop.find('span', class_='vw-dynprops-item-attr').text.strip(':')
        value = prop.find('span', class_='vw-dynprops-item-val').text.strip()

        # Assign values based on attribute
        if attribute == 'Кузов':
            car_body = value
        elif attribute == 'Год':
            car_year = value
        elif attribute == 'Коробка':
            car_transmission = value
        elif attribute == 'Пробег':
            car_mileage = value
        elif attribute == 'Объём':
            car_volume = value
        elif attribute == 'Привод':
            car_drive = value
        elif attribute == 'Руль':
            car_steering_wheel = value
        elif attribute == 'Растаможен':
            car_customs_clearance = value
        elif attribute == 'Цвет':
            car_color = value

    total_dict = {"Name": car_name,
                  "Body": car_body,
                  "Year": car_year,
                  "Price": car_price,
                  "Transmission": car_transmission,
                  "Mileage": car_mileage,
                  "Volume": car_volume,
                  "Car Drive": car_drive, 
                  "Steering wheel": car_steering_wheel, 
                  "Customs Clearance": car_customs_clearance,
                  "Color": car_color,
                  "url": url}
    total_dict.update(about_dict)
    return total_dict""")

clean_data = pd.read_csv("csv_files/clean_data.csv")
st.dataframe(clean_data)

