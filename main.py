import json
import string
import webbrowser
import word_matrix as wma
from wikicrawler import get_link
import PySimpleGUI as sg


def create_window(matches=None):
    head1 = [sg.Text("Welcome to ScienceSearch")]
    head2 = [sg.Button("Search"), sg.Input('', key='Input1')]
    layout = [head1, head2]
    if matches is not None:
        for name, link in matches:
            layout.append([sg.Text(name + ":")])
            layout.append([sg.Text("\t" + link)])
    window = sg.Window('Simple data entry window', layout, finalize=True)
    window['Input1'].bind("<Return>", "_Enter")
    return window


def gui():
    sg.theme('DarkGrey')
    sg.set_options(font=("Impact", 16))
    window = create_window()
    while True:
        event, values = window.read()
        if event == "Search" or event == "Input1" + "_Enter":
            text = values["Input1"]
            matches = get_matches(text)
            window.close()
            window = create_window(matches)
        if event == sg.WIN_CLOSED:
            break
    window.close()
    return


def get_matches(text: string):
    best = wm.compare(text, 4)
    return [(sites[i], get_link(sites[i])) for i in best]


if __name__ == '__main__':
    wm = wma.WordMatrix()
    wm.read()
    with open('jsons/sites.json', 'r') as read_file:
        sites = json.load(read_file)
    gui()
