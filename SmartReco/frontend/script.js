async function getRecommendations() {
  const query = document.getElementById("query").value;
  const output = document.getElementById("results");

  if (!query) {
    alert("Please enter a product name or description!");
    return;
  }

  output.innerHTML = "🔍 Searching recommendations...";

  try {
    const response = await fetch(`http://127.0.0.1:8000/recommend?query=${encodeURIComponent(query)}&top_k=3`, {
      method: "POST",
      headers: { Accept: "application/json" },
    });

    if (!response.ok) throw new Error("Server error");

    const data = await response.json();
    const results = data.recommendations || [];

    if (results.length === 0) {
      output.innerHTML = "⚠️ No recommendations found.";
      return;
    }

    output.innerHTML = results
      .map(
        (p) => `
        <div class="card">
          <img src="${p.image_url}" alt="${p.name}" />
          <h3>${p.name}</h3>
          <p><strong>Brand:</strong> ${p.brand}</p>
          <p>${p.description}</p>
          <p><strong>Category:</strong> ${p.category}</p>
          <p><strong>Material:</strong> ${p.material} | <strong>Color:</strong> ${p.color}</p>
          <p><strong>Price:</strong> ₹${p.price}</p>
        </div>
      `
      )
      .join("");
  } catch (err) {
    console.error(err);
    output.innerHTML = "⚠️ Unable to fetch recommendations.";
  }
}
