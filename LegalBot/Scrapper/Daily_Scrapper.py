import time
import os
import requests
import fitz
import tempfile
import re

from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager

class daily_scrapper:
    def __init__(self, Scrapper_Folder_Path):
        self.Scrapper_Folder_Path = Scrapper_Folder_Path
        self.Scraped_Content_Folder_Path = os.path.join(self.Scrapper_Folder_Path, 'Scraped_Content')
        self.New_Content_Folder_Path = os.path.join(self.Scrapper_Folder_Path, 'New_Content')
        self.current_year = time.gmtime().tm_year
        self.URL_DAILY_UPDATE = {'Uk' : 'https://www.legislation.gov.uk/new/uk',
                                'Wales' : 'https://www.legislation.gov.uk/new/wales',
                                'Scotland' : 'https://www.legislation.gov.uk/new/scotland',
                                'Northern Ireland' : 'https://www.legislation.gov.uk/new/ni',}
        
    def verify_daily_update(self, driver):
        '''
        A function that chechs if a new published daily update exists in a tab
        '''
        h5_content = driver.find_element(By.CLASS_NAME, 'p_content').find_element(By.TAG_NAME, 'h5').text
        if h5_content == 'Nothing published on this date':
            return False
        return True

    def extract_content_from_pdf(self, pdf_url):
        '''
        A function that extracts the content from a pdf file if the title is a PDF.
        '''
        response = requests.get(f'{pdf_url}')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
            temp_pdf_path = temp_pdf_file.name
            response = requests.get(pdf_url)
            temp_pdf_file.write(response.content)

        # Open the temporary PDF file and extract text content
        pdf_document = fitz.open(temp_pdf_path)
        text_content = ''
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text_content += page.get_text()
        pdf_document.close()

        return text_content

    def extract_content(self, driver, title_url):
        '''
        Function that extracts the content from a tab. NOte: This is same as in Scrapper.ipynb
        '''
        
        driver.get(title_url)
        time.sleep(1)
        Title_Content_Div = driver.find_element(By.CSS_SELECTOR, 'div.legToc')
        NavBar = Title_Content_Div.find_element(By.ID, 'legSubNav')
        NavBarLists = NavBar.find_elements(By.TAG_NAME, 'li')
        ContentTab = NavBarLists[1] #It may be clickable or not. If not, the media type is PDF not text
        Content_Link_Tag = None
        try:  #If Not PDF
            Content_Link_Tag = ContentTab.find_element(By.TAG_NAME, 'a')
        except:
            Content_Link_Tag = None
        
        if Content_Link_Tag != None:
            Content_Link_Tag_Href = Content_Link_Tag.get_attribute('href')
            driver.get(Content_Link_Tag_Href)
            time.sleep(1)
            '''Now get the content'''
            '''Multiple Pages of the content page'''
            Page_Number = 1
            All_Provisions_Text = ''
            while True:
                Content_Box = driver.find_element(By.ID, 'content')
                Content_Text = Content_Box.find_element(By.ID, 'viewLegContents').find_element(By.CLASS_NAME, 'LegSnippet')
                page_Text = Content_Text.text
                All_Provisions_Text += page_Text
                print(f'Page Number: {Page_Number}')
                
                '''Now check for button'''
                Button_Panel = driver.find_element(By.CLASS_NAME, 'prevNextNav')
                try:
                    Next_Button = Button_Panel.find_element(By.TAG_NAME, 'ul').find_elements(By.TAG_NAME, 'li')[-1].find_element(By.TAG_NAME, 'a')
                    print(f'Next Button found: {Next_Button.text}')
                    try:
                        Next_Button.click()
                        time.sleep(1)
                        Page_Number += 1
                    except:
                        print(f'You are probably on the very last Provision page')
                        print(f'Provision Page Number: {Page_Number}')
                        break
                except:
                    print(f'No Next Button Found - Last Provision Page')
                    print(f'Provision Page Number: {Page_Number}')
                    break
            return All_Provisions_Text
        
        elif Content_Link_Tag == None:
            Tag_PDF_href = driver.find_element(By.CSS_SELECTOR, 'div.LegSnippet').find_element(By.TAG_NAME, 'a').get_attribute('href')
            pdf_content = self.extract_content_from_pdf(Tag_PDF_href)
            return pdf_content

    def get_daily_update(self, driver, url):
        '''
        A function that extracts the daily update from a tab.
        What is a tab? Each country's new title's website is a tab.
        '''
        driver.get(url)
        time.sleep(1)
        
        if self.verify_daily_update(driver) == True:
            New_Titles = {}
            Titles_Href_List = []
            Title_Name_List = []

            Content_div = driver.find_element(By.CLASS_NAME, 'p_content')
            Legislation_Name = Content_div.find_element(By.TAG_NAME, 'h5').text
            Title_URLS = Content_div.find_elements(By.TAG_NAME, 'h6')

            for idxNewTitle, title in enumerate(Title_URLS):
                href = title.find_element(By.TAG_NAME, 'a').get_attribute('href')
                name = title.text.split('-')[-1].strip()
                Titles_Href_List.append(href)
                Title_Name_List.append(name)

            if len(Titles_Href_List) > 0:
                New_Titles[f'{Legislation_Name}'] = dict(zip(Title_Name_List, Titles_Href_List))
                return New_Titles
            else:
                return None
            
    def create_dirs(self, path):
        '''
        A function that creates directories if they don't exist.
        '''
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            os.makedirs(abs_path)
            
    def replace_slashes(self, filename):
        # Replace backslashes and forward slashes with unique placeholders
        filename = filename.replace('\\', '__BS__') #Back Slash
        filename = filename.replace('/', '__FS__') #Front Slash
        return filename

    def restore_slashes(self, filename):
        # Restore the placeholders back to original slashes
        filename = filename.replace('__BS__', '\\') #Back Slash
        filename = filename.replace('__FS__', '/') #Front Slash
        return filename

    def append_problem(self, title_name, title_path, problem):
        self.create_dirs(
            path=f'{self.Scrapper_Folder_Path}/Problematic_Titles'
        )

        #create a file called "problematic_files.txt" and int save the titlename and its path and its reason
        with open(f'{self.Scrapper_Folder_Path}/Problematic_Titles/problematic_files.txt', 'a') as f:
            f.write(title_name + '\n')
            f.write(title_path + '\n')
            f.write(f'{problem}' + '\n')
            f.write('---\n')
        f.close()
            
    def get_legislation_type(self, legislation_name):
        '''
        A function that returns the list of countries and the legislation type in which the legislation is in.
        '''
        All_Content_Folder = self.Scraped_Content_Folder_Path
        dict_web_structure = {}
        dict_web_structure[str(legislation_name)] = [{'Country': Country, 'LegislationType': LegislationType} for idxCountry, Country in enumerate(os.listdir(All_Content_Folder)) for idxLegislationType, LegislationType in enumerate(os.listdir(f'{All_Content_Folder}/{Country}')) if legislation_name in os.listdir(f'{All_Content_Folder}/{Country}/{LegislationType}')]
        return dict_web_structure
            
    def main(self):
        '''
        A function that extracts the daily update from all tabs. This is the main driver function for the daily update scrapper class.
        '''
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        
        # driver_path = ChromeDriverManager().install()
        # print(f'driver_path: {driver_path}') 
        # driver = webdriver.Chrome(service=ChromeService(executable_path=driver_path))
        driver = webdriver.Chrome(options=options) #For docker use
        
        for Country_Key in self.URL_DAILY_UPDATE.keys():
            New_Titles = self.get_daily_update(driver, self.URL_DAILY_UPDATE[Country_Key])
            if New_Titles != {} and New_Titles is not None:
                for idxLegislation, legislation_name in enumerate(New_Titles.keys()):
                    for title_name, title_url in New_Titles[legislation_name].items():
                        '''Now since we have the title url, we can extract the title data'''
                        legislation_type = self.get_legislation_type(legislation_name)
                        try:
                            Title_Data_Content = self.extract_content(driver, title_url)
                        except:
                            yield None

                        for item in legislation_type[legislation_name]:
                            key_country = item['Country']
                            key_legislation_type = item['LegislationType']

                            txt_file_path = f'{self.New_Content_Folder_Path}/{key_country}/{key_legislation_type}/{legislation_name}/{self.current_year}'
                            validated_file_name = self.replace_slashes(title_name)
                            validated_file_name = os.path.join(txt_file_path, validated_file_name+'.txt')

                            self.create_dirs(path=txt_file_path)

                            try:
                                with open(f'{validated_file_name}', 'w') as f:
                                    f.write(Title_Data_Content)
                                f.close()
                                print(f'Successfully created file: {validated_file_name}')
                            except Exception as e:
                                error_message = str(e)
                                print(f'Error Message: {error_message}')
                                self.append_problem(title_name=title_name, 
                                                title_path=validated_file_name, 
                                                problem=error_message)
                                print(f'Failed to create file: {validated_file_name}')
                                
                            if key_country == 'UK':
                                key_country = 'Uk'
                        
                            yield {
                                'Text': Title_Data_Content,
                                'Meta Data' : {
                                    'Country': key_country,
                                    'LegislationType': key_legislation_type,
                                    'Legislation': legislation_name,
                                    'Year': str(self.current_year),
                                    'Title': title_name
                                    }
                                }
                        