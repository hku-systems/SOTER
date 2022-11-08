
XXX="Cronus"
XXX_nosrpc="Cronus(sRPC disabled)"

# marker
markers     = [".", "v", "*", "s", "d", "+"]
lines       = ["-", ":", "--", "-."]
hatches     = ["/////", ".", "\\", "*","x","-"]

# colors      = ['#FF6666', '#FFFF66', '#99CC66']
# colors      = ['#FF9966', '#FFFFCC', '#99CC99', "#336699"]
colors        = ['r', 'b', 'yellow', 'purple']
colors        = ['#6D8764', '#647687', '#76608A', '#A0522D']
colors        = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:olive']
# colors      = ['#ffcc99', '#ffff99', '#99cc99']
# colors      = ['#CC6600', '#CCCC33', '#336699']

LEGEND_SIZE=20
LABEL_SIZE=22
TICK_SIZE=20
MARKER_SIZE=10

bar_format_args = {
    'zorder': 2,
    'color':'white'
}

LINE_RED="#b0004c"
LINE_BLUE="#0089a4"
LINE_GREEN="#89a400"
LINE_PURPLE="tab:purple"

MARKER_A = "o"
MARKER_B = "s"
MARKER_C = "^"
MARKER_D = "p"

LINE_COLOR              =       [LINE_RED, LINE_BLUE, LINE_GREEN, LINE_PURPLE]
LINE_MARKER             =       [MARKER_A, MARKER_B, MARKER_C, MARKER_D]

LINE_WIDTH=2

ALPHA = .8

line_format_args = {
    'linewidth':LINE_WIDTH, 
    'markersize':MARKER_SIZE, 
    'alpha':ALPHA, 
    'mec':'black'
}

IS_DEBUG=False