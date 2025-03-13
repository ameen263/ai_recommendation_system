// src/components/ui/card.js
import React from 'react';

const Card = ({ children, title }) => {
  return (
    <article className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
      {title && <h2 className="text-xl font-bold mb-2">{title}</h2>}
      <div>{children}</div>
    </article>
  );
};

export const CardHeader = ({ children }) => (
    <header className="mb-2">{children}</header>
);

export const CardContent = ({ children }) => (
    <section>{children}</section>
);

export const CardTitle = ({ children }) => (
  <h2 className="text-lg font-bold mb-2">{children}</h2>
);

export default Card;