const grid = document.getElementById("grid");
const output = document.getElementById("output");

let mode = "brush";
let mouseDown = false;

document.getElementById("brush").onclick = () => mode = "brush";
document.getElementById("erase").onclick = () => mode = "erase";

// detect mouse hold
document.body.onmousedown = () => mouseDown = true;
document.body.onmouseup = () => mouseDown = false;

const cells = [];

// create 64 cells
for (let i = 0; i < 63 ; i++) {

    const cell = document.createElement("div");
    cell.classList.add("cell");

    // single click draw
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

// export array
document.getElementById("export").onclick = () => {

    const data = cells.map(cell =>
        cell.classList.contains("filled") ? 1 : 0
    );

    let formatted = "[\n";

    for (let i = 0; i < 63; i += 7) {
        formatted += "  " + data.slice(i, i + 7).join(",") + ",\n";
    }

    formatted += "]";

    output.textContent = formatted;
};