const grid = document.getElementById("grid");

const output = document.getElementById("output");

const result = document.getElementById("result");

const weightsDiv = document.getElementById("weights");


let mode = "brush";

let mouseDown = false;


// buttons
document.getElementById("brush").onclick = () => mode = "brush";

document.getElementById("erase").onclick = () => mode = "erase";


// detect mouse hold
document.body.onmousedown = () => mouseDown = true;

document.body.onmouseup = () => mouseDown = false;


const cells = [];


// create 63 cells
for (let i = 0; i < 63; i++) {

    const cell = document.createElement("div");

    cell.classList.add("cell");


    // single click
    cell.addEventListener("click", paintCell);


    // drag draw
    cell.addEventListener("mouseover", () => {

        if (mouseDown) {
            paintCell();
        }
    });


    function paintCell() {

        if (mode === "brush") {

            cell.classList.add("filled");

        } else {

            cell.classList.remove("filled");
        }
    }


    grid.appendChild(cell);

    cells.push(cell);
}


// =========================
// EXPORT ARRAY
// =========================

document.getElementById("export").onclick = () => {

    const data = getPixelArray();

    let formatted = "[\n";

    for (let i = 0; i < 63; i += 7) {

        formatted += "  " + data.slice(i, i + 7).join(",") + ",\n";
    }

    formatted += "]";

    output.textContent = formatted;
};


// =========================
// GET PIXEL ARRAY
// =========================

function getPixelArray() {

    return cells.map(cell =>

        cell.classList.contains("filled") ? 1 : 0
    );
}


// =========================
// PREDICT BUTTON
// =========================

document.getElementById("predict").onclick = async () => {

    const pixels = getPixelArray();

    console.log("Sending pixels:", pixels);


    try {

        const response = await fetch("http://127.0.0.1:5000/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                pixels: pixels
            })
        });


        const data = await response.json();

        console.log("Backend response:", data);


        // probabilities
        const sixProb = (data.six * 100).toFixed(2);

        const sevenProb = (data.seven * 100).toFixed(2);

        const noneProb = (data.none * 100).toFixed(2);


        // prediction
        let prediction = "None";

        if (data.six > data.seven && data.six > data.none) {
            prediction = "6";
        }

        if (data.seven > data.six && data.seven > data.none) {
            prediction = "7";
        }


        // show result
        result.innerHTML = `

            <h2>Prediction: ${prediction}</h2>

            <p>6: ${sixProb}%</p>

            <p>7: ${sevenProb}%</p>

            <p>None: ${noneProb}%</p>
        `;


        // show weights
        weightsDiv.innerHTML = `

            <h3>6 MODEL</h3>

            <p><b>Bias:</b> ${data.six_bias}</p>

            <p><b>Weights:</b> ${data.six_weights}</p>


            <hr>


            <h3>7 MODEL</h3>

            <p><b>Bias:</b> ${data.seven_bias}</p>

            <p><b>Weights:</b> ${data.seven_weights}</p>


            <hr>


            <h3>NONE MODEL</h3>

            <p><b>Bias:</b> ${data.none_bias}</p>

            <p><b>Weights:</b> ${data.none_weights}</p>
        `;

    } catch (error) {

        console.error("ERROR:", error);

        result.innerHTML = `
            <p style="color:red;">
                Failed to connect to backend.
            </p>
        `;
    }
};