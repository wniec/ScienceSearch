import json
import string
import webbrowser
import word_matrix as wma
from wikicrawler import get_link
import PySimpleGUI as sG


def create_window(matches=None):
    head1 = [sG.Text("Welcome to ScienceSearch")]
    head2 = [sG.Button("Search"), sG.Input('', key='Input1')]
    layout = [head1, head2]
    if matches is not None:
        for name, link in matches:
            layout.append([sG.Text(name, tooltip=link, enable_events=True, key=f'URL {link}')])
    window = sG.Window('ScienceSearch', layout, finalize=True)
    window['Input1'].bind("<Return>", "_Enter")
    return window


def gui():
    sG.theme('DarkGrey')
    sG.set_options(font=("Montserrat", 16))
    window = create_window()
    while True:
        event, values = window.read()
        if event == "Search" or event == "Input1" + "_Enter":
            text = values["Input1"]
            matches = get_matches(text)
            window.close()
            window = create_window(matches)
        if event == sG.WIN_CLOSED:
            break
        elif event.startswith("URL "):
            url = event.split(' ')[1]
            webbrowser.open(url)
    window.close()
    return


def get_matches(text: string):
    best = wm.compare(text, 4)
    return [(sites[i], get_link(sites[i])) for i in best]


if __name__ == '__main__':
    wm = wma.WordMatrix()
    wm.read()
    wm.lower_rank()
    wm.save()
    #with open('saved/sites.json', 'r') as read_file:
    #    sites = json.load(read_file)
    #gui()
