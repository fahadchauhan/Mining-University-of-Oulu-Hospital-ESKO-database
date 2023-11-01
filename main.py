import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Canvas, PhotoImage

import pandas as pd
import matplotlib.pyplot as plt
from googletrans import Translator
import time
from tqdm import tqdm
import numpy as np
from scipy.stats import skew, kurtosis
from wordcloud import WordCloud, STOPWORDS
import empath
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import umls_api
import string
import requests
import mysql.connector
import spacy
import re


def plot_bar_graph(counts, text1, text2):
    plt.figure(figsize=(8, 6))
    ax = counts.plot(kind='bar', color=['lightcoral', 'skyblue'])
    plt.title('Proportion of Patients with Narrative Text')
    plt.xticks([0, 1], [text1, text2])
    plt.xlabel('Narrative Status')
    plt.ylabel('Number of Patients')

    # Add labels to the bars
    for i in range(len(counts)):
        plt.text(i, counts[i], str(counts[i]), ha='center', va='bottom')
    plt.show()


class MiningUniHospData:
    def __init__(self, root):

        self.output_label = None
        self.submit_button = None
        self.input_entry = None
        self.entry = None
        self.df = pd.read_csv("syoparap_merkinta_otsikko.csv", parse_dates=['insert_date_utc'], dayfirst=True)
        self.df_with_narratives = pd.read_csv('translated_data.csv', parse_dates=['insert_date_utc'], dayfirst=True)
        self.df_with_medical_terms = pd.read_csv('translated_data_with_medical_terms.csv',
                                                 parse_dates=['insert_date_utc'],
                                                 dayfirst=True)
        self.df_recommendations_with_medical_terms = pd.read_csv('recommendations_with_medical_terms.csv')
        self.category_df = pd.read_csv('empath_categories_on_translated_narratives.csv')
        self.category_df_recommendations = pd.read_csv('empath_categories_on_recommendations.csv')

        self.semantic_types_tui = ['T007', 'T004', 'T005',
                                   'T018', 'T019', 'T020', 'T023', 'T024', 'T025', 'T026', 'T028',
                                   'T200',
                                   'T121', 'T122', 'T125', 'T126', 'T127', 'T129', 'T192', 'T130', 'T131', 'T114',
                                   'T116',
                                   'T197', 'T196', 'T031', 'T168',
                                   'T022', 'T030', 'T029', 'T085', 'T086', 'T087', 'T088',
                                   'T034', 'T184',
                                   'T032', 'T201',
                                   'T054', 'T055',
                                   'T056',
                                   'T058', 'T059', 'T060', 'T061', 'T063',
                                   'T039', 'T040', 'T041', 'T042', 'T043', 'T044', 'T045',
                                   'T046', 'T047', 'T048', 'T191', 'T049', 'T050',
                                   'T037',
                                   'T154',
                                   'T163']

        self.root = root
        self.root.title("Mining University of Oulu Hospital ESKO database")

        self.create_widgets()

    def create_widgets(self):
        padding = 10
        button_frame = tk.Frame(self.root)
        button_frame.pack(padx=padding, pady=padding)

        self.proportion_of_patients_with_narrative_text = tk.Button(button_frame,
                                                                    text="Find proportion of patients with narrative text",
                                                                    command=self.find_proportion_of_patients_with_narrative_text)
        self.proportion_of_patients_with_narrative_text.pack(pady=padding)

        self.stats_of_translated_narratives = tk.Button(button_frame, text="Find Statistics on Translated Narratives",
                                                        command=self.find_stats_of_translated_narratives)
        self.stats_of_translated_narratives.pack(pady=padding)

        self.wordcloud_on_translated_narratives = tk.Button(button_frame,
                                                            text="Display WordCloud of Translated Narratives",
                                                            command=self.find_wordcloud_on_translated_narrative)
        self.wordcloud_on_translated_narratives.pack(pady=padding)

        self.empath_categories_on_translated_narratives = tk.Button(button_frame,
                                                                    text="Display Top Empath Categories of Translated Narratives",
                                                                    command=self.find_empath_categories_on_translated_narratives)
        self.empath_categories_on_translated_narratives.pack(pady=padding)

        self.stats_of_medical_terms_in_translated_narratives = tk.Button(button_frame,
                                                                         text="Display stats of medical terms in Translated Narratives",
                                                                         command=self.find_stats_of_medical_terms_in_translated_narratives)
        self.stats_of_medical_terms_in_translated_narratives.pack(pady=padding)

        self.medical_terms_from_input_text = tk.Button(button_frame,
                                                       text="Find Medical terms by inputting text",
                                                       command=self.find_medical_terms_from_input_text)
        self.medical_terms_from_input_text.pack(pady=padding)

        self.correlation_between_medical_vocabulary_and_number_of_words = tk.Button(button_frame,
                                                                                    text="Find correlation between medical vocabulary and number of words",
                                                                                    command=self.find_correlation_between_medical_vocabulary_and_number_of_words)
        self.correlation_between_medical_vocabulary_and_number_of_words.pack(pady=padding)

        self.wordCloud_of_recommendations = tk.Button(button_frame,
                                                      text="Display WordCloud of Recommendations",
                                                      command=self.display_wordcloud_of_recommendations)
        self.wordCloud_of_recommendations.pack(pady=padding)

        self.proportion_of_the_medical_vocabulary_used_in_recommendation = tk.Button(button_frame,
                                                                                     text="Find Proportion of the medical vocabulary used in recommendation",
                                                                                     command=self.find_proportion_of_the_medical_vocabulary_used_in_recommendation)
        self.proportion_of_the_medical_vocabulary_used_in_recommendation.pack(pady=padding)

        self.empath_categories_in_recommendations = tk.Button(button_frame,
                                                              text="Find empath categories in recommendations",
                                                              command=self.find_empath_categories_in_recommendations)
        self.empath_categories_in_recommendations.pack(pady=padding)

    def find_proportion_of_patients_with_narrative_text(self):
        self.df['narrative_present'] = self.df['narratiiviteksti'].apply(
            lambda x: isinstance(x, str) and x.strip() != '').astype(
            bool)
        counts = self.df['narrative_present'].value_counts()

        title = 'Proportion of Patients with Narrative Text'
        xlabel = 'Narrative Status'
        ylabel = 'Number of Patients'
        text1 = 'Narrative Absent'
        text2 = 'Narrative Present'

        self.display_bar_graph(counts, title, xlabel, ylabel, text1, text2)

    def display_bar_graph(self, counts, title, xlabel, ylabel, text1, text2):
        # Create bar graph
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts, color=['lightcoral', 'skyblue'])

        # Add labels to the bars
        for i, v in enumerate(counts):
            ax.text(i, v, str(v), ha='center', va='bottom')

        # Customize plot
        plt.title(title)
        plt.xticks(counts.index, [text1, text2])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Create a new window to display the graph
        pop_up = tk.Toplevel(self.root)
        pop_up.title("Narrative Text Presence")

        # Embed the graph in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=pop_up)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def find_stats_of_translated_narratives(self):
        # Assuming you have a DataFrame df with a 'translated_narrative' column
        narratives = self.df_with_narratives['translated_narrative'].dropna()  # Remove any NaN values

        # Function to count words in a text
        def count_words(text):
            words = text.split()
            return len(words)

        # Calculate the number of words in each narrative
        word_counts = narratives.apply(count_words)

        # Calculate the mean, standard deviation, skewness, and kurtosis
        mean_word_count = word_counts.mean()
        std_word_count = word_counts.std()
        skewness = skew(word_counts)
        kurt = kurtosis(word_counts)

        # Create a new window to display the results
        pop_up = tk.Toplevel(self.root)
        pop_up.title("Statistics on Translated Narratives")

        # Create a label to display the results
        stats_label = tk.Label(pop_up, text=f"Mean Word Count: {mean_word_count:.2f}\n"
                                            f"Standard Deviation of Word Count: {std_word_count:.2f}\n"
                                            f"Skewness: {skewness:.2f}\n"
                                            f"Kurtosis: {kurt:.2f}\n")
        stats_label.pack()

    def find_wordcloud_on_translated_narrative(self):
        # Combine all the translated narratives into a single string
        narratives_text = ' '.join(self.df_with_narratives['translated_narrative'].dropna())

        # Create a WordCloud object
        wordcloud = WordCloud(
            background_color='white',
            stopwords=set(STOPWORDS),  # You can add more stopwords if needed
            width=800,
            height=800,
            min_font_size=10
        ).generate(narratives_text)

        # Save the WordCloud to an image file
        wordcloud.to_file("wordcloud_translated_narratives.png")

        title = "WordCloud on Translated Narratives"
        self.display_wordcloud("wordcloud_translated_narratives.png", title)

    def display_wordcloud(self, image_file, title):
        # Create a new Tkinter window
        wordcloud_window = tk.Toplevel(self.root)
        wordcloud_window.title(title)

        # Load the image file and display it on a canvas
        img = tk.PhotoImage(file=image_file)
        canv = tk.Canvas(wordcloud_window, width=img.width(), height=img.height())
        canv.create_image(0, 0, anchor='nw', image=img)
        canv.pack()

        # Display the WordCloud pop-up window
        wordcloud_window.mainloop()

    def find_empath_categories_on_translated_narratives(self):
        category_proportions = self.category_df.mean()
        filtered_categories = category_proportions[category_proportions > 0.001]

        title = 'Proportion of > 0.001 Categories in Narratives'
        xlabel = 'Categories'
        ylabel = 'Proportion'

        # Create a new Tkinter window
        pop_up = tk.Toplevel(self.root)
        pop_up.title("Empath Categories in Translated Narratives")

        # Plot the Bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        filtered_categories.sort_values(ascending=False).plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(filtered_categories.index, rotation=90)
        plt.subplots_adjust(bottom=0.3)

        # Embed the plot in the new window
        canvas = FigureCanvasTkAgg(fig, master=pop_up)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Display the plot in a new window
        pop_up.mainloop()

    def find_stats_of_medical_terms_in_translated_narratives(self):
        # Calculate statistics
        mean_medical_terms = self.df_with_medical_terms['medical_term_count'].mean()
        std_medical_terms = self.df_with_medical_terms['medical_term_count'].std()
        skewness_medical_terms = self.df_with_medical_terms['medical_term_count'].skew()
        kurtosis_medical_terms = self.df_with_medical_terms['medical_term_count'].kurtosis()

        self.display_bar_plot_of_medical_terms(['Mean', 'Standard Deviation'], [mean_medical_terms, std_medical_terms],
                                               'Mean and Standard Deviation of Medical Terms')
        self.display_bar_plot_of_medical_terms(['Skewness', 'Kurtosis'],
                                               [skewness_medical_terms, kurtosis_medical_terms],
                                               'Skewness and Kurtosis of Medical Terms')

    def display_bar_plot_of_medical_terms(self, x_labels, y_values, title):
        plt.figure(figsize=(8, 6))
        plt.bar(x_labels, y_values)
        plt.xlabel('Statistics')
        plt.ylabel('Value')
        plt.title(title)
        plt.tight_layout()

        # Create a new window to display the graph
        pop_up = tk.Toplevel(self.root)
        pop_up.title(title)

        # Embed the graph in the Tkinter window
        canvas = FigureCanvasTkAgg(plt.gcf(), master=pop_up)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def find_medical_terms_from_input_text(self):
        input_window = tk.Toplevel(self.root)
        input_window.title("Enter Text")

        self.input_entry = tk.Entry(input_window)
        self.input_entry.pack()

        self.submit_button = tk.Button(input_window, text="Submit", command=self.find_medical_terms)
        self.submit_button.pack()

        self.output_label = tk.Label(input_window, text="")
        self.output_label.pack()

    def create_connection(self):
        db_config = {
            "host": "localhost",
            "user": "root",
            "password": "12545",
            "database": "umls_2022aa",
        }

        try:
            # Establish a connection
            self.conn = mysql.connector.connect(**db_config)

            # Create a cursor
            self.cursor = self.conn.cursor()
        except mysql.connector.Error as err:
            print(f"Error: {err}")

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

    def match_word(self, word):

        try:
            sql_query = """
            SELECT DISTINCT b.cui, b.str, a.tui
            FROM mrsty a
            JOIN mrconso b ON a.cui = b.cui
            WHERE b.str = %s
            AND a.tui IN ({})""".format(', '.join(['%s'] * len(self.semantic_types_tui)))

            # Parameters for the SQL query
            query_parameters = [word] + self.semantic_types_tui

            # Execute the query with parameters
            self.cursor.execute(sql_query, query_parameters)
            results = self.cursor.fetchall()

            if len(results) > 0:
                return True
            else:
                return False

        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return False

    def find_medical_terms(self):
        self.create_connection()

        medical_terms = []
        temp_terms = []

        text = self.input_entry.get()
        # Tokenize and preprocess the text
        text = text.lower()
        text = ' '.join(text.split())
        text = text.replace('\n', ' ')
        words = word_tokenize(text)
        words = [word for word in words if word not in string.punctuation]
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        words = [word for word in words if not word.isnumeric()]

        # Initialize a variable to count medical terms
        medical_term_count = 0

        # Search for UMLS terms in the text
        for word in words:
            term_matches = self.match_word(word)
            if term_matches:
                medical_term_count += 1
                temp_terms.append(word)
            # else:
            #     print(f'term: {word} not found')

        medical_terms.append(temp_terms)

        self.close_connection()

        self.display_medical_terms_from_input_text(medical_terms, medical_term_count)

    def display_medical_terms_from_input_text(self, medical_terms, medical_term_count):
        output_text = f"Medical Terms Count: {medical_term_count}\nMedical Terms: {medical_terms[0]}"
        self.output_label.config(text=f"{output_text}")

    def find_correlation_between_medical_vocabulary_and_number_of_words(self):
        self.df_with_medical_terms['medical_terms'] = self.df_with_medical_terms['medical_terms'].apply(
            eval)  # only run, if you have just loaded the dataframe from csv, cuz it will be in string format
        medical_terms = self.df_with_medical_terms['medical_terms'].apply(lambda x: x[0]).explode()

        # Count the frequency of each medical term
        medical_terms_frequency = medical_terms.value_counts().to_dict()

        # Select the most frequent medical terms and their counts
        top_medical_terms = list(medical_terms_frequency.keys())

        # Filter medical terms with a frequency greater than 10
        filtered_top_terms = [term for term in top_medical_terms if medical_terms_frequency[term] > 2200]

        # Create a DataFrame for the filtered terms
        filtered_data = self.df_with_medical_terms[
            self.df_with_medical_terms['medical_terms'].apply(
                lambda x: any(term in x[0] for term in filtered_top_terms))]

        # Calculate total texts, average size, and standard deviation for each filtered term
        filtered_total_texts = []
        filtered_average_sizes = []
        filtered_std_deviations = []

        for term in tqdm(filtered_top_terms):
            term_texts = filtered_data[filtered_data['medical_terms'].apply(lambda x: term in x[0])]

            filtered_total_texts.append(len(term_texts))
            filtered_average_sizes.append(term_texts['translated_narrative'].str.split().apply(len).mean())
            filtered_std_deviations.append(term_texts['translated_narrative'].str.split().apply(len).std())

        # def display_histogram(self):
        # Create a figure and plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(filtered_top_terms, filtered_total_texts, alpha=0.5, label='Total Texts')
        ax.set_xlabel('Medical Terms')
        ax.set_ylabel('Total Texts')
        ax.set_xticklabels(filtered_top_terms, rotation=45)

        ax2 = ax.twinx()
        ax2.plot(filtered_top_terms, filtered_average_sizes, 'g', marker='o', label='Average Size')
        ax2.plot(filtered_top_terms, filtered_std_deviations, 'r', marker='s', label='Standard Deviation')
        ax2.set_ylabel('Average Size and Standard Deviation')

        ax.set_title('Medical Vocabulary vs. Text Statistics (Frequency > 2000)')
        ax.legend(loc='upper right')

        # Create a Tkinter window to display the plot
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Medical Vocabulary and Text Statistics")

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def display_wordcloud_of_recommendations(self):
        recommendations_string = ' '.join(self.df_recommendations_with_medical_terms['Recommendations'])

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(recommendations_string)

        # Save the WordCloud to an image file
        wordcloud.to_file("wordcloud_recommendations.png")
        title = "WordCloud on Recommendations"
        self.display_wordcloud("wordcloud_recommendations.png", title)

    def find_proportion_of_the_medical_vocabulary_used_in_recommendation(self):
        recommendations = self.df_recommendations_with_medical_terms['Recommendations']
        medical_terms = self.df_recommendations_with_medical_terms['Medical Terms']

        # Initialize a counter for recommendations with medical terms
        recommendations_with_medical_count = 0
        total_recommendations = 0

        # Iterate through each recommendation and check for the presence of medical terms
        for rec, terms in zip(recommendations, medical_terms):
            total_recommendations += len(rec)
            if any(term in rec for term in terms):
                recommendations_with_medical_count += 1

        # Calculate the proportion
        # total_recommendations = len(recommendations)
        proportion = (recommendations_with_medical_count / total_recommendations) * 100

        # def display_recommendations_chart(self):
        #     self.calculate_proportion()  # Call the function to perform the calculations

        # Create a figure and plot
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['Recommendations with Medical Terms', 'Recommendations without Medical Terms']
        counts = [recommendations_with_medical_count, total_recommendations - recommendations_with_medical_count]

        bars = ax.bar(labels, counts, color=['blue', 'gray'])
        ax.set_ylabel('Count')
        ax.set_title('Proportion of Medical Vocabulary in Recommendations')

        # Add labels with the total count on each bar
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2 - 0.06, bar.get_height() + 300, str(count), fontsize=12,
                    color='black')

        # Create a Tkinter window to display the plot
        chart_window = tk.Toplevel(self.root)
        chart_window.title("Recommendations Analysis")

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def find_empath_categories_in_recommendations(self):
        new_window = tk.Toplevel(self.root)
        new_window.title("Histogram")

        category_proportions = self.category_df_recommendations.mean()
        filtered_categories = category_proportions[category_proportions > 0.002]

        # Plot the histogram
        fig = plt.Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        bars = ax.bar(filtered_categories.index, filtered_categories, color='skyblue')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Proportion')
        ax.set_title('Proportion of > 0.001 Categories in Narratives')

        # Set the rotation of x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

        # Customize y-axis ticks with added padding
        yticks = ax.get_yticks()
        padding = (yticks[-1] - yticks[-2]) * 0.7  # Adjust the factor as needed for desired padding
        new_yticks = list(yticks) + [yticks[-1] + padding]
        new_yticklabels = [str(int(y)) for y in new_yticks]

        ax.set_yticks(new_yticks)
        ax.set_yticklabels(new_yticklabels)

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)


def main():
    root = tk.Tk()
    app = MiningUniHospData(root)
    root.mainloop()


if __name__ == "__main__":
    main()
