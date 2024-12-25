import globals from 'globals'
import js from '@eslint/js'

/** @type {import('eslint').Linter.Config[]} */
export default [
  {
    files: ['**/*.js'],
    languageOptions: {
      sourceType: 'commonjs',
      globals: { ...globals.node, ...globals.browser },
    },
    rules: {
      eqeqeq: 'error',

      /*
        Google Style Guide
        Got a good chunk of the rules from google/eslint-config-google repository.
        link: https://github.com/google/eslint-config-google/blob/master/index.js
      */

      // Best Practices
      curly: ['error', 'multi-line'],
      'guard-for-in': 'error',
      'no-caller': 'error',
      'no-extend-native': 'error',
      'no-extra-bind': 'error',
      'no-invalid-this': 'error',
      'no-multi-str': 'error',
      'no-new-wrappers': 'error',
      'prefer-promise-reject-errors': 'error',

      // Stylistic Issues
      camelcase: ['error', { properties: 'never' }],
      'no-array-constructor': 'error',
      'no-new-object': 'error',
      'one-var': [
        'error',
        {
          var: 'never',
          let: 'never',
          const: 'never',
        },
      ],

      // ECMAScript 6
      'no-var': 'error',
      'prefer-const': ['error', { destructuring: 'all' }],
      'prefer-rest-params': 'error',
      'prefer-spread': 'error',
    },
  },
  js.configs.recommended,
]
