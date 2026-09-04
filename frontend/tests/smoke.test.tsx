import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "../src/App";

describe("App", () => {
  it("renders its shell", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /tidescout/i })).toBeInTheDocument();
  });
});
