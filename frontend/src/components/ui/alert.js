// src/components/ui/alert.js
import React from 'react';

const Alert = ({ variant = 'info', children, ...props }) => {
  let colorClass = '';
  switch (variant) {
    case 'info':
      colorClass = 'bg-blue-100 text-blue-800';
      break;
    case 'success':
      colorClass = 'bg-green-100 text-green-800';
      break;
    case 'warning':
      colorClass = 'bg-yellow-100 text-yellow-800';
      break;
    case 'destructive':
      colorClass = 'bg-red-100 text-red-800';
      break;
    default:
      colorClass = 'bg-gray-100 text-gray-800'; // Default color
  }

  return (
    <div className={`p-4 rounded-lg border border-gray-300 ${colorClass}`} {...props} role="alert">
      {children}
    </div>
  );
};


export const AlertDescription = ({ children }) => (
  <p className="text-sm font-normal mt-1">{children}</p>
);

export default Alert;