import ForceGraph from "https://esm.sh/force-graph";
import { select } from "https://esm.sh/d3-selection";
import { extent, min, max } from "https://esm.sh/d3-array";
import { brush } from "https://esm.sh/d3-brush@3";
import { forceManyBody } from "https://esm.sh/d3";
import { scaleLinear, scaleOrdinal, scaleIdentity } from "https://esm.sh/d3-scale";
import { schemeCategory10 } from "https://esm.sh/d3-scale-chromatic";
import RBush from 'https://cdn.jsdelivr.net/npm/rbush/+esm';

let default_width = 800;
let default_height = 500;

class MyRBush extends RBush {
    toBBox(node) { return { id: node.id, minX: node.x, minY: node.y, maxX: node.x, maxY: node.y }; }
    compareMinX(a, b) { return a.x - b.x; }
    compareMinY(a, b) { return a.y - b.y; }
}

let local_selected_ids = []
let default_repulsion = 1;
let default_node_scale = 1;
let colour_scale_type = "";
let plot;
let tree;
let default_node_size = 5;
let colour_feature = undefined;
let node_size_feature = undefined;
let colour_scale;

function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

const create_rtree = (data) => {
    tree = new MyRBush()
    tree.load(data)
}

const get_feature_type = (value) => {
    if (typeof value === 'number') return 'numeric'; // Check for numeric
    if (typeof value === 'string' && value.startsWith('#')) return 'hash_string'; // Check for strings starting with #
    return 'categorical'; // Default to categorical if not numeric or hash_string
}

const ua = navigator.userAgent.toLowerCase();

const isMac = ua.includes('macintosh');
const isWindows = ua.includes('windows');
const isLinux = ua.includes('linux');

const create_node_canvas_object = (plot, node_scale, node_size_feature) => {
    return plot.nodeCanvasObject((node, ctx) => {
        console.log(node_size_feature)
        let radius;

        if ( node_size_feature == undefined || node_size_feature == "") {
            radius = default_node_size * node_scale
        } else {
            radius = node[node_size_feature] * node_scale
        }
        const isLocalSelected = local_selected_ids.includes(node.id);
        const fillColour = colour_scale(node[colour_feature]);
        // const radius = node[node_size_feature] * node_scale;

        // Draw filled circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
        ctx.fillStyle = fillColour;
        ctx.fill();

        // Draw red stroke if node is locally selected
        if (isLocalSelected) {
            ctx.lineWidth = radius/5;
            ctx.strokeStyle = 'red';
            ctx.stroke();
        } else {
            ctx.lineWidth = radius/10;
            ctx.strokeStyle = 'lightgrey';
            ctx.stroke();
        }
    });
};

function render({ model, el }) {
    const debouncedSaveChanges = debounce(() => model.save_changes(), 300);

    const create_plot = (data) => {
        return ForceGraph()(el)
            .width(width)
            .height(height)
            .graphData(data)
            // .d3Force('charge',forceManyBody().strength(-repulsion))
            .cooldownTime(5000)
            .warmupTicks(10)
            .nodeLabel('label')
            .d3AlphaDecay(0.001)
            .minZoom(0.001)
            .nodeCanvasObjectMode(() => 'replace')
            .onEngineStop(() => {
                plot.zoomToFit(400);
                create_rtree(data["nodes"]);
            });
    }

    const data = model.get("data");
    let repulsion = model.get("repulsion") || default_repulsion;
    let node_scale = model.get("node_scale") || default_node_scale;
    let width = model.get("width") || default_width;
    let height = model.get("height") || default_height;
    colour_scale_type = model.get("colour_scale_type");
    colour_feature = model.get("colour_feature");
    node_size_feature = model.get("node_size_feature");
    let global_selected_ids = model.get("selected_ids");
    
    plot = create_plot(data);

    const update_repulsion = () => {
        repulsion = model.get("repulsion");
        plot.d3Force('charge', forceManyBody().strength(-repulsion));
        plot.d3ReheatSimulation();
    }
    model.on("change:repulsion", update_repulsion)

    const update_node_scale = () => {
        node_scale = model.get("node_scale");
        create_node_canvas_object(plot, node_scale, node_size_feature);
        plot.d3ReheatSimulation();
    }
    model.on("change:node_scale", update_node_scale)

    const update_node_size_feature = () => {
        node_size_feature = model.get("node_size_feature");
        create_node_canvas_object(plot, node_scale, node_size_feature);
    }
    model.on("change:node_size_feature", update_node_size_feature);

    const update_colour_feature = () => {
        colour_feature = model.get("colour_feature");
        let feature_values = data.nodes.map(node => node[colour_feature])
        let feature_type = get_feature_type(feature_values[0])
        if ( feature_type == "numeric" ) {
            if ( colour_scale_type == "diverging" ) {
                colour_scale = scaleLinear().domain([min(feature_values), 0, max(feature_values)]).range(["#0000FFBF","white","#FF0000BF"])
            } else {
                colour_scale = scaleLinear().domain(extent(feature_values)).range(["#FFFF0080", "#0000FF80"])
            }
        } else if ( feature_type == "categorical" ) {
            colour_scale = scaleOrdinal().domain(extent(feature_values)).range(schemeCategory10)
        } else if ( feature_type == "hash_string" ) {
            colour_scale = scaleIdentity()
        } else {
            colour_scale = undefined
        }

        if (colour_scale_type === "diverging") {
            // Sort nodes by absolute distance from zero
            data.nodes.sort((a, b) => {
                const aVal = Math.abs(a[colour_feature]);
                const bVal = Math.abs(b[colour_feature]);
                return aVal - bVal;
            });
        }

        create_node_canvas_object(plot, node_scale, node_size_feature);
    }
    model.on("change:colour_feature", update_colour_feature)

    const update_colour_scale_type = () => {
        colour_feature = model.get("colour_feature");
        colour_scale_type = model.get("colour_scale_type")
        let feature_values = data.nodes.map(node => node[colour_feature])
        let feature_type = get_feature_type(feature_values[0])
        if ( feature_type == "numeric" ) {
            if ( colour_scale_type == "diverging" ) {
                colour_scale = scaleLinear().domain([min(feature_values), 0, max(feature_values)]).range(["#0000FFBF","white","#FF0000BF"])
            } else {
                colour_scale = scaleLinear().domain(extent(feature_values)).range(["#FFFF0080", "#0000FF80"])
            }
        } else if ( feature_type == "categorical" ) {
            colour_scale = scaleOrdinal().domain(extent(feature_values)).range(schemeCategory10)
        } else if ( feature_type == "hash_string" ) {
            colour_scale = scaleIdentity()
        } else {
            colour_scale = () => "#4682B480"; // Default colour
        }

        if (colour_scale_type === "diverging") {
            // Sort nodes by absolute distance from zero
            data.nodes.sort((a, b) => {
                const aVal = Math.abs(a[colour_feature]);
                const bVal = Math.abs(b[colour_feature]);
                return aVal - bVal;
            });
        }

        create_node_canvas_object(plot, node_scale, node_size_feature);
    }
    model.on("change:colour_scale_type", update_colour_scale_type)

    model.on("change:selected_ids", () => {
        global_selected_ids = model.get("selected_ids");
        plot.nodeColor((d) =>
            global_selected_ids.includes(d.id)
                ? "red"
            : local_selected_ids.includes(d.id)
                ? "rgba(255,0,0,0.5)"
                : d["colour_hex"],
        );
    });

    update_node_scale();
    update_colour_feature();
    update_node_size_feature();

    let brush_active = false;

    let container = el.querySelector(".force-graph-container")

    // Create an overlay for brushing
    let overlay = select(container)
        .append("svg")
        .attr("id", "overlay")
        .style("position", "absolute")
        .style("top", 0)
        .style("left", 0)
        .style("width", "100%")
        .style("height", "100%")
        .style("pointer-events", "none")

    let activate_brush = () => {
        brush_active = true;
        el.style.pointerEvents = "none"
        overlay.style.pointerEvents = "auto"

        if (!overlay.select("#brush_group").empty()) return;
        overlay.insert("g", ":first-child")
            .attr("id", "brush_group")
            .attr("class", "brush")
            .call(my_brush)
    }
    let disactivate_brush = () => {
        brush_active = false;
        el.style.pointerEvents = "auto"
        overlay.style.pointerEvents = "none"

        overlay.select("#brush_group").remove();
    }

    let brushed = (event) => {
        if (event.selection) {
            brush_active = true;
            let [[x0_screen, y0_screen], [x1_screen, y1_screen]] =
                event.selection;
            let corner0 = plot.screen2GraphCoords(x0_screen, y0_screen);
            let corner1 = plot.screen2GraphCoords(x1_screen, y1_screen);
            let bbox = {
                minX: Math.min(corner0.x, corner1.x),
                minY: Math.min(corner0.y, corner1.y),
                maxX: Math.max(corner0.x, corner1.x),
                maxY: Math.max(corner0.y, corner1.y)
            };
            let selectedNodes = tree.search(bbox)
            local_selected_ids = selectedNodes.map(node => node.id);
            model.set("selected_ids", local_selected_ids);
            debouncedSaveChanges();
        }
    };

    let my_brush = brush()
        .filter((event) => {
            const modifierKey = isMac ? event.metaKey : event.ctrlKey;
            console.log(modifierKey, event.button, event.target.__data__.type, event);
            return (
                event.button == 0 && // Ignore mouse buttons other than left-click
                (modifierKey || ["selection", "s", "e", "n", "w"].includes(event.target.__data__.type))
            )
        })
        .extent([[0, 0], [width, height]])
        .on("start brush end", brushed);

    window.addEventListener("keydown", (e) => {
        if (e.metaKey && !brush_active) { activate_brush() }
    })
    window.addEventListener("dblclick", (e) => {
        if (brush_active) { disactivate_brush() }
    })

}

export default { render };