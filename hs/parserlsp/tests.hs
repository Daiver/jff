import Test.HUnit

import Lexer
import Parser
import ParserLsp

lexerTest1 = TestCase (assertEqual "" [(1, 1, "123")] 
                                    (tokenize "123"))

lexerTest2 = TestCase (assertEqual "" [(1, 1, "123"), (1, 4, "("), (1, 5, "15"), (1, 7, ")")] 
                                    (tokenize "123(15)"))

lexerTest3 = TestCase (assertEqual "" [(1, 1, "123"), (1, 4, "("), (1, 5, "15"), (1, 7, " "), (1, 8, "+"), (1, 9, " "), (1, 10, "6"), (1, 11, ")")] 
                                    (tokenize "123(15 + 6)"))

lexerTest4 = TestCase (assertEqual "" [(1, 1, "123"), (1, 4, "("), (1, 5, "15"), (1, 7, " "), (1, 8, "+"), (1, 9, "6"), (1, 10, ")")] 
                                    (tokenize "123(15 +6)"))

lexerTest5 = TestCase (assertEqual "" [(1,1,"  "), (1, 3, "123"), (1, 6, "("), (1,7,")"), (1, 8, " "), (1, 9, "["), (1,10,"]"), (1, 11, "\n"), (2, 1, "1")] 
                                    (tokenize "  123() []\n1"))

lexerTest6 = TestCase (assertEqual "" [(1,1,"  "), (1, 3, "123"), (1, 6, "("), (1, 7, "--"), (1, 9, ")"), (1, 10, " "), (1, 11, "["), (1,12,"]"), (1, 13, "\n"), (2, 1, "1")] 
                                    (tokenize "  123(--) []\n1"))

lexerTest7 = TestCase (assertEqual "" [(1,1,"  "), (1, 3, "123"), (1, 6, "("), (1, 7, "--"), (1, 9, ")"), (1, 10, " "), (1, 11, "["), (1,12,"]"), (1, 13, "\n"), (2, 1, "\n"), (3, 1, "1"), (3, 2, " "), (3, 3, "lol")] 
                                    (tokenize "  123(--) []\n\n1 lol"))

lexerTest8 = TestCase (assertEqual "" [(1,1,"  "), (1, 3, "123"), (1, 6, "("), (1, 7, "--"), (1, 9, ")"), (1, 10, " "), (1, 11, "["), (1,12,"]"), (1, 13, "\n"), (2, 1, "\n"), (3, 1, "1"), (3, 2, " "), (3, 3, "lol"), (3, 6, "\n")] 
                                    (tokenize "  123(--) []\n\n1 lol\n"))

lexerTest9 = TestCase (assertEqual "" [(1,1, "11"), (1, 4, "+")] (removeUselessTokens [(1,1, "11"), (1, 3, " "), (1, 4, "+")]))

lexerTest10 = TestCase (assertEqual "" [(1,1, "11"), (1, 4, "+")] (removeUselessTokens [(1,1, "11"), (1, 3, " "), (1, 4, "+"), (1,1,"\n"), (1,1,"   ")]))

lexerTest11 = TestCase (assertEqual "" [(1,1, "11"), (1, 4, "+")] (removeUselessTokens [(1,1, "\n"), (1,1,"\n"), (1,1,"      \t"), (1,1, "11"), (1, 3, " "), (1, 4, "+"), (1,1,"\n"), (1,1,"   ")]))

lexerTest12 = TestCase
            (assertEqual ""
                [(3, 1, "2"), (3, 4, "+"), (3, 6, "2")]
                (removeUselessTokens . tokenize $ "\n      \n2  + 2     \n\n   \n")
            )

simpleParser1 = Parser $ \inp -> do
    char '['
    lst <- many digit
    char ']'
    return lst

parserTest1 = TestCase 
                (assertEqual ""
                    [1,2,3,4]
                    (papply simpleParser1 "[12345]")
                )

tests = TestList [
                    TestLabel "Lexer test 1" lexerTest1,
                    TestLabel "Lexer test 2" lexerTest2,
                    TestLabel "Lexer test 3" lexerTest3,
                    TestLabel "Lexer test 4" lexerTest4,
                    TestLabel "Lexer test 5" lexerTest5,
                    TestLabel "Lexer test 6" lexerTest6,
                    TestLabel "Lexer test 7" lexerTest7,
                    TestLabel "Lexer test 8" lexerTest8,
                    TestLabel "Lexer test 9" lexerTest9,
                    TestLabel "Lexer test 10" lexerTest10,
                    TestLabel "Lexer test 11" lexerTest11,
                    TestLabel "Lexer test 12" lexerTest12,
                    TestLabel "Parser test 1" parserTest1
                ]

main = runTestTT tests
