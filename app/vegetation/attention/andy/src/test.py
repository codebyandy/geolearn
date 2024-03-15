from inference import analysis, create_map

dict = {"TEST": ["24-03-03_TEST/0/model_450", 450],
        "TEST1": ["24-03-03_TEST/0/model_450", 450]}

# metrics, fig = analysis(dict)

create_map(dict["TEST"])