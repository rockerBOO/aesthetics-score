const x = (endpoint) => {
  fetch(endpoint)
    .then((resp) => resp.json())
    .then((v) => {
      const blocks = v
        .sort((a, b) => b.score - a.score)
        .map(({ file, score }) => {
          const div = document.createElement("div");
          const img = document.createElement("img");

          img.src = `http://localhost:3456/img/${file}`;
          const file_block = document.createElement("div");
          file_block.classList.add("file");
          file_block.innerText = file;

          const score_block = document.createElement("div");
          score_block.innerText = score.toPrecision(6);
          score_block.classList.add("score");

          div.appendChild(img);
          div.appendChild(file_block);
          div.appendChild(score_block);

          div.classList.add("block");

          return div;
        });

      const main = document.querySelector("#main");

      main.replaceChildren(...blocks);
    });
};
import("https://cdn.skypack.dev/lodash-es/debounce").then((v) => {
	const debounce = v.default;
  import("https://cdn.skypack.dev/dat.gui").then(({ GUI }) => {
    const state = {
      min: 0,
      max: 10,
    };
		const gui = new GUI({ closed: true, outerWidth: "2vw" });
    gui
      .add(state, "min", 0, 10, 1)
      .onChange(
        debounce(() => x(`filter/x.json?min=${state.min}&max=${state.max}`), 400)
      );
    gui
      .add(state, "max", 0, 10, 1)
      .onChange(
        debounce(() => x(`filter/x.json?min=${state.min}&max=${state.max}`), 400)
      );

		x(`filter/x.json?min=0&max=10`)
  });
});
