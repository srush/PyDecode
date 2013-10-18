from beamer_animation import latex_lib 
#import pyplot



latex_lib.write_diagram("dot", r"\node [circle] {};")
latex_lib.write_animation("dots", 
                          [r"\node [circle, fill=%s] {};"%color 
                           for color in ["red", "blue", "green"]])
# write_plot()
