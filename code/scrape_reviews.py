import os
import time
import pandas as pd
from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        # Parameters
        checkin_date = '2025-01-27'
        checkout_date = '2025-01-31'
        url = f'https://www.booking.com/searchresults.html?ss=Spain&checkin={checkin_date}&checkout={checkout_date}&group_adults=2&no_rooms=1&group_children=0'

        # Launch browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # Wait for the content to load
        page.wait_for_selector('div[data-testid="property-card"]')

        # Initialize hotel list and total count
        hotels_list = []
        total_hotel_count = 0

        while True:
            # Retrieve hotels on the current page
            hotels = page.query_selector_all('div[data-testid="property-card"]')
            print(f"Found {len(hotels)} hotels on the current page.")

            for hotel in hotels:
                hotel_name = hotel.query_selector('div[data-testid="title"]')
                hotel_price = hotel.query_selector('span[data-testid="price-and-discounted-price"]')
                review_score_div = hotel.query_selector('div[data-testid="review-score"]')

                # Initialize variables
                score = 'No Score'
                reviews_count = 'No Reviews'

                # Extract review score and review count
                if review_score_div:
                    score_elements = review_score_div.query_selector_all('div')
                    if score_elements and len(score_elements) > 0:
                        raw_score = score_elements[0].inner_text()
                        score = raw_score.split()[-1]  # Extract the numeric part

                    # Extract review count
                    reviews_count_div = review_score_div.query_selector('div.abf093bdfe.f45d8e4c32.d935416c47')
                    if reviews_count_div:
                        reviews_count = reviews_count_div.inner_text().split()[0]  # Extract the number

                # Add hotel details to the list
                hotel_dict = {
                    'hotel': hotel_name.inner_text() if hotel_name else 'No hotel name available',
                    'price': hotel_price.inner_text() if hotel_price else 'No price available',
                    'score': score,
                    'reviews count': reviews_count
                }
                hotels_list.append(hotel_dict)

                # Update total hotel count
                total_hotel_count += 1

                # Stop if 3,000 hotels are collected
                if total_hotel_count >= 3000:
                    print("Collected 3,000 hotels. Stopping.")
                    break

            # Print the total number of hotels accumulated in hotels_list
            print(f"Total hotels scraped so far: {total_hotel_count}")

            # Stop if 3,000 hotels are collected
            if total_hotel_count >= 3000:
                break

            # Detect and click the "Load more results" button
            load_more_button = page.query_selector('button.a83ed08757 span.e4adce92df')
            if load_more_button:
                print("Clicking 'Load more results' button...")
                page.evaluate("document.querySelector('button.a83ed08757').click()")
                time.sleep(3)  # Wait for new results to load
            else:
                print("No more 'Load more results' button found. Exiting.")
                break

        # Ensure 'data' directory exists
        if not os.path.exists('../data'):
            os.makedirs('../data')

        # Save to CSV
        df = pd.DataFrame(hotels_list)
        df.to_csv('../data/hotels_list_3000.csv', index=False)
        print("Data saved to data/hotels_list_3000.csv")

        # Close browser
        browser.close()
        print("Scraping completed and browser closed.")

if __name__ == '__main__':
    main()
