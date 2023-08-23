const create = (tag) => document.createElement(tag);
const $ = (v) => document.querySelector(v);

const render = (v) => {
  const blocks = v
    .sort((a, b) => b.score - a.score)
    .map(({ file, score }) => {
      const div = create("div");
      const img = create("img");

      img.src = `/img/${file}`;
      const file_block = create("div");
      file_block.classList.add("file");
      file_block.innerText = file;

      const score_block = create("div");
      score_block.innerText = score.toPrecision(6);
      score_block.classList.add("score");

      div.appendChild(img);
      div.appendChild(file_block);
      div.appendChild(score_block);

      div.classList.add("block");

      return div;
    });

  $("#main").replaceChildren(...blocks);
};

const fetch_results = async (endpoint) => {
  return fetch(endpoint).then((resp) => resp.json());
};

import("https://cdn.skypack.dev/lodash-es/debounce").then((v) => {
  const debounce = v.default;
  import("https://cdn.skypack.dev/dat.gui").then(({ GUI }) => {
    const state = {
      min: 0,
      max: 10,
      file: "",
    };
    const gui = new GUI({
      closed: true,
      outerWidth: "minmax(2vw, 75vw)",
      load: state,
    });
    const folder = gui.addFolder("score");
    folder.add(state, "min", 0, 10, 0.1).onChange(
      debounce(() => {
        fetch_results(`filter/x.json?min=${state.min}&max=${state.max}`).then(
          render
        );
      }, 400)
    );
    folder.add(state, "max", 0, 10, 1).onChange(
      debounce(() => {
        fetch_results(`filter/x.json?min=${state.min}&max=${state.max}`).then(
          render
        );
      }, 400)
    );

    gui.add(state, "file").onFinishChange(
      debounce(() => {
        fetch_results(`./a.json?file=${state.file}`).then(render);
      })
    );

    fetch_results(`filter/x.json?min=0&max=10`).then(render);
  });
});
